import re

from peregrine.misc import subclass_where
from peregrine.simulator.mixture.constants import Tref
from peregrine.simulator.mixture.reaction import BaseReaction
from peregrine.simulator.mixture.species import Species
from peregrine.simulator.mixture.parser import units


class BaseParser:
    name = None
    # the file suffixes this format is read from, lower case
    suffixes = ()
    defunits = None

    def __init__(self, path, species=None):
        self._build(self.load(path), species, path)

    # the file as the dict Cantera's yaml of it loads to; every format is read through that layout
    def load(self, path):
        raise NotImplementedError

    def _build(self, data, species, path):
        if species is None:
            if "phases" not in data:
                msg = f"{path} has no phases section; give the species list"
                raise ValueError(msg)
            species = list(data["phases"][0]["species"])
        lookup = {sp["name"]: sp for sp in data.get("species", [])}
        # the file's own elements (custom or overriding weights) win
        weights = dict(Species.atomwts)
        for el in data.get("elements", []):
            weights[el["symbol"]] = float(el["atomic-weight"])
        self.species = {}
        for name in species:
            if name not in lookup:
                raise KeyError(f"Species '{name}' not found in {path}")
            self.species[name] = self._species(lookup[name], weights)

        self.reactions = []
        if data.get("reactions"):
            fileunits = {**self.defunits, **data.get("units", {})}
            rxnunits = units.ReactionUnits(fileunits)
            for raw in data["reactions"]:
                rxn = self._reaction(raw, rxnunits)
                unknown = set(rxn["reactants"]) | set(rxn["products"])
                unknown |= set(rxn["efficiencies"])
                unknown -= set(self.species)
                if unknown:
                    raise KeyError(
                        f"Reaction '{rxn['equation']}' uses species not in the phase: {sorted(unknown)}"
                    )
                self.reactions.append(rxn)

    @staticmethod
    def _bool(sect, key, default=False):
        v = sect.get(key, default)
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "on")
        return bool(v)

    @staticmethod
    def _side(side):
        species, hasm = {}, False
        side = re.sub(r"\(\s*\+[^)]+\)", "", side).strip()
        for term in side.split("+"):
            term = term.strip()
            if not term:
                continue
            if term == "M":
                if hasm:
                    raise ValueError(
                        "Multiple generic third-body colliders 'M' are not supported"
                    )
                hasm = True
                continue
            m = re.match(r"^(\d+\.?\d*)\s+(.+)$", term)
            coeff, name = (float(m.group(1)), m.group(2).strip()) if m else (1.0, term)
            species[name] = species.get(name, 0.0) + coeff
        return species, hasm

    @staticmethod
    def _equation(equation):
        equation = equation.split("#")[0].strip()
        found = re.findall(r"\(\s*\+([^)]+)\)", equation)
        colliders = {c.strip() for c in found}
        if len(colliders) > 1:
            raise ValueError(f"Inconsistent falloff colliders in '{equation}'")
        collider = colliders.pop() if colliders else None
        if "<=>" in equation:
            lhs, rhs = equation.split("<=>")
            reversible = True
        elif "=>" in equation:
            lhs, rhs = equation.split("=>")
            reversible = False
        else:
            raise ValueError(f"Cannot parse equation: '{equation}'")
        reactants, lhsm = BaseParser._side(lhs)
        products, rhsm = BaseParser._side(rhs)
        if lhsm != rhsm:
            msg = f"Third body M must appear on both sides: '{equation}'"
            raise ValueError(msg)
        return reactants, products, reversible, lhsm, collider

    @staticmethod
    def _reaction(raw, rxnunits):
        equation = raw["equation"]
        parsed = BaseParser._equation(equation)
        reactants, products, reversible, hasm, collider = parsed
        rawtype = raw.get("type", None)
        effs = {k: float(v) for k, v in raw.get("efficiencies", {}).items()}
        defaultEff = float(raw.get("default-efficiency", 1.0))

        if rawtype == "Arrhenius":
            rawtype = "elementary"
        elif rawtype == "three-body-Arrhenius":
            rawtype = "three-body"
        elif rawtype in ("Lindemann", "Troe"):
            rawtype = "falloff"

        hastroe = "Troe" in raw
        if rawtype == "falloff":
            rtype = "troe" if hastroe else "lindemann"
            if collider is None:
                raise ValueError(
                    f"Falloff reaction requires a (+M) or (+species) collider: '{equation}'"
                )
            if collider != "M":
                # explicit collider: zero default efficiency, unit collider efficiency
                if "default-efficiency" in raw and defaultEff != 0.0:
                    raise ValueError(
                        f"Invalid default efficiency for explicit collider: '{equation}'"
                    )
                if effs and (len(effs) != 1 or collider not in effs):
                    raise ValueError(
                        f"Incompatible third-body collider definitions: '{equation}'"
                    )
                defaultEff = 0.0
                effs = effs or {collider: 1.0}
        elif rawtype == "three-body" or (rawtype is None and hasm):
            rtype = "three-body"
            if collider is not None:
                raise ValueError(
                    f"'(+{collider})' collider requires type: falloff: '{equation}'"
                )
            if not hasm:
                # explicit-species collider: one unit removed from each side
                if len(effs) != 1:
                    raise ValueError(
                        f"Third-body definition requires a single-species efficiency: '{equation}'"
                    )
                (sp,) = effs
                for side in (reactants, products):
                    if side.get(sp, 0) < 1:
                        raise ValueError(
                            f"Third-body collider '{sp}' must appear on both sides: '{equation}'"
                        )
                    if side[sp] == 1:
                        del side[sp]
                    else:
                        side[sp] -= 1
                defaultEff = 0.0
        elif rawtype in (None, "elementary"):
            rtype = "elementary"
            if hasm or collider is not None:
                raise ValueError(f"Elementary reaction with a third body: '{equation}'")
            if "efficiencies" in raw or "default-efficiency" in raw:
                raise ValueError(
                    f"Efficiencies on a reaction without third bodies: '{equation}'"
                )
        else:
            msg = f"Unsupported reaction type '{rawtype}': '{equation}'"
            raise ValueError(msg)

        orders = {sp: float(v) for sp, v in raw.get("orders", {}).items()}
        if orders:
            if reversible:
                raise ValueError(
                    f"Reaction orders may only be given for irreversible reactions: '{equation}'"
                )
            nonreactant = any(sp not in reactants for sp in orders)
            if nonreactant and not BaseParser._bool(raw, "nonreactant-orders"):
                raise ValueError(
                    f"Reaction order specified for non-reactant species: '{equation}'"
                )
            negative = any(v < 0 for v in orders.values())
            if negative and not BaseParser._bool(raw, "negative-orders"):
                msg = f"Negative reaction order specified: '{equation}'"
                raise ValueError(msg)
        # forward concentration exponents: orders override stoichiometry
        fwd = dict(reactants)
        fwd.update(orders)
        kind = subclass_where(BaseReaction, name=rtype)
        order = sum(fwd.values()) + kind.bodyorder

        rxn = {
            "equation": equation,
            "type": rtype,
            "reversible": reversible,
            "duplicate": BaseParser._bool(raw, "duplicate"),
            "reactants": reactants,
            "products": products,
            "fwd": fwd,
            "efficiencies": effs,
            "defeff": defaultEff,
        }
        if kind.falloff:
            high = raw["high-P-rate-constant"]
            rxn["rate"] = rxnunits.arrhenius(high, order)
            low = raw["low-P-rate-constant"]
            rxn["lowrate"] = rxnunits.arrhenius(low, order + 1)
        else:
            rxn["rate"] = rxnunits.arrhenius(raw["rate-constant"], order)
        if hastroe:
            troe = {"T2": 0.0, **raw["Troe"]}
            rxn["troe"] = [float(troe[k]) for k in ("A", "T3", "T1", "T2")]
        return rxn

    @staticmethod
    def _species(raw, weights):
        sp = {}
        if "composition" in raw:
            sp["comp"] = {k: float(v) for k, v in raw["composition"].items()}
            sp["MW"] = Species.molwt(sp["comp"], weights)
        if "thermo" in raw and raw["thermo"]["model"] == "constant-cp":
            thermo = raw["thermo"]
            # J/(kg K) from the molar value with its unit
            cp0 = units.molar(thermo["cp0"])
            sp["cp0"] = cp0 / sp["MW"]
            # the enthalpy at the reference temperature, J/kmol, from the file's h0 at its T0, zero at 298.15 K unless given
            T0 = float(str(thermo.get("T0", "298.15 K")).split()[0])
            h0 = units.molar(thermo["h0"]) if "h0" in thermo else 0.0
            sp["dHf298"] = h0 + cp0 * (Tref - T0)
        elif "thermo" in raw:
            thermo = raw["thermo"]
            model = thermo["model"]
            if model.upper() != "NASA7":
                msg = f"Unsupported thermo model {model} for {raw['name']}"
                raise ValueError(msg)
            ranges = [float(r) for r in thermo["temperature-ranges"]]
            data = [list(map(float, c)) for c in thermo["data"]]
            # a single fit serves both ranges
            low, high = (data[0], data[0]) if len(data) == 1 else data[:2]
            Tmid = ranges[1] if len(data) > 1 else ranges[-1]
            # source data with the range it is good for; it gets refit to the case's
            Trange = [ranges[0], Tmid, ranges[-1]]
            sp["NASA7"] = {"T-range": Trange, "high": high, "low": low}
        if "transport" in raw:
            tr = raw["transport"]
            sp["geometry"] = tr["geometry"]
            for key, ours in (
                ("well-depth", "well"),
                ("diameter", "diam"),
                ("dipole", "dipole"),
                ("polarizability", "polarize"),
            ):
                unit, table = units.transport[key]
                sp[ours] = units.si(tr.get(key, 0.0), unit, table)
            sp["zrot"] = float(tr.get("rotational-relaxation", 0.0))
            # absent is unstated, not zero; a zero would shadow the library's value
            if "acentric-factor" in tr:
                sp["acentric"] = float(tr["acentric-factor"])
        return sp
