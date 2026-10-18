import re

from peregrine.simulator.mixture.parser import units
from peregrine.simulator.mixture.parser.base import BaseParser


class ChemkinParser(BaseParser):
    name = "chemkin"
    suffixes = (".inp", ".dat", ".ck", ".mech")
    # CGS, moles and cal/mole, which the REACTIONS line only ever departs from
    defunits = {
        "length": "cm", "quantity": "mol", "activation-energy": "cal/mol"
    }
    # a section starts at a line whose first word is one of these
    sectnames = {
        "ELEM": "ELEMENTS",
        "ELEMENTS": "ELEMENTS",
        "SPEC": "SPECIES",
        "SPECIES": "SPECIES",
        "THER": "THERMO",
        "THERMO": "THERMO",
        "REAC": "REACTIONS",
        "REACTIONS": "REACTIONS",
        "TRAN": "TRANSPORT",
        "TRANSPORT": "TRANSPORT",
    }
    # auxiliary keywords this reader refuses; any other word is an efficiency and has to name a species
    refused = (
        "HIGH",
        "SRI",
        "REV",
        "PLOG",
        "CHEB",
        "TCHEB",
        "PCHEB",
        "FORD",
        "RORD",
        "UNITS",
        "LT",
        "RLT",
        "JAN",
        "FIT1",
        "HV",
        "TDEP",
        "EXCI",
        "MOME",
        "XSMI",
        "COLLEFF",
    )

    def load(self, path):
        with open(path, errors="replace") as f:
            return self.translate(f.read())

    @classmethod
    def translate(cls, text):
        sections = cls._sections(text)
        empty = ("", [])
        elements, weights = cls._elements(sections.get("ELEMENTS", empty)[1])
        names = " ".join(sections.get("SPECIES", empty)[1]).split()
        if not names:
            raise ValueError("the mechanism declares no SPECIES")
        carried = [k for k in ("THERMO", "TRANSPORT") if k in sections]
        if carried:
            raise ValueError(
                f"the mechanism carries {' and '.join(carried)} data: convert it to "
                "Cantera yaml with ck2yaml and give that"
            )
        fileunits, reactions = {}, []
        if "REACTIONS" in sections:
            keywords, lines = sections["REACTIONS"]
            fileunits = cls._units(keywords)
            reactions = cls._reactions(lines, set(names))
        phase = {
            "name": "gas",
            "thermo": "ideal-gas",
            "elements": elements,
            "species": names,
        }
        species = [{"name": name} for name in names]
        data = {"units": fileunits, "phases": [phase], "species": species}
        if weights:
            data["elements"] = weights
        if reactions:
            phase["kinetics"] = "gas"
            data["reactions"] = reactions
        return data

    @classmethod
    def _sections(cls, text):
        sections, current = {}, None
        for raw in text.splitlines():
            line = raw.split("!", 1)[0].rstrip()
            words = line.split()
            if not words:
                continue
            key = words[0].upper()
            if key in cls.sectnames:
                current = cls.sectnames[key]
                rest = line.split(None, 1)[1] if len(words) > 1 else ""
                inline = current in ("ELEMENTS", "SPECIES")
                content = [rest] if inline and rest else []
                sections[current] = ("" if inline else rest, content)
            elif key == "END":
                current = None
            elif current is not None:
                sections[current][1].append(line)
        return sections

    @staticmethod
    def _float(text):
        return float(text.strip().replace("D", "E").replace("d", "e"))

    @classmethod
    def _is_float(cls, text):
        try:
            cls._float(text)
        except ValueError:
            return False
        return True

    @classmethod
    def _elements(cls, lines):
        symbols, weights = [], []
        text = re.sub(r"\s*/\s*([^/\s]+)\s*/", r"/\1/", " ".join(lines))
        for token in text.split():
            symbol, _, weight = token.partition("/")
            symbol = symbol.capitalize()
            symbols.append(symbol)
            if weight:
                weight = cls._float(weight.rstrip("/"))
                weights.append({"symbol": symbol, "atomic-weight": weight})
        return symbols, weights

    @staticmethod
    def _units(keywords):
        out = {}
        for word in keywords.upper().split():
            name = units.factor(units.chemkin, word)
            key = "quantity" if name in units.quantity else "activation-energy"
            out[key] = name
        return out

    @classmethod
    def _reactions(cls, lines, names):
        reactions = []
        for line in lines:
            m = re.match(r"^(.*=.*?)\s+(\S+)\s+(\S+)\s+(\S+)$", line.strip())
            if m and all(map(cls._is_float, m.groups()[1:])):
                arrhenius = [cls._float(x) for x in m.groups()[1:]]
                equation = m.group(1)
                reactions.append(cls._rxnline(equation, arrhenius, names))
            elif reactions:
                cls._auxline(line, reactions[-1], names)
            else:
                raise ValueError(f"{line.strip()!r} is not a reaction")
        for rxn in reactions:
            eq = rxn["equation"]
            falloff = rxn.get("type") == "falloff"
            if falloff and "low-P-rate-constant" not in rxn:
                raise ValueError(f"{eq}: a falloff reaction needs LOW")
            if not falloff and ("low-P-rate-constant" in rxn or "Troe" in rxn):
                raise ValueError(f"{eq}: LOW and TROE need a (+M) collider")
        return reactions

    @classmethod
    def _rxnline(cls, equation, arrhenius, names):
        reversible = "<=>" in equation or "=>" not in equation
        sides, colliders, thirdbody = [], set(), []
        for side in re.split(r"<=>|=>|=", equation, maxsplit=1):
            found = re.finditer(r"\(\s*\+\s*([^)\s]+)\s*\)", side)
            colliders |= {m.group(1) for m in found}
            side = re.sub(r"\(\s*\+[^)]*\)", "", side)
            terms, hasm = cls._terms(side, names, equation)
            sides.append(terms)
            thirdbody.append(hasm)
        if len(colliders) > 1:
            msg = f"{equation}: one falloff collider, not {sorted(colliders)}"
            raise ValueError(msg)
        if thirdbody[0] != thirdbody[1]:
            raise ValueError(f"{equation}: M has to be on both sides")
        collider = colliders.pop() if colliders else None
        text = [
            " + ".join(f"{c:g} {n}" if c != 1 else n for n, c in terms)
            for terms in sides
        ]
        if thirdbody[0]:
            text = [t + " + M" for t in text]
        if collider:
            text = [t + f" (+{collider})" for t in text]
        rate = dict(zip(("A", "b", "Ea"), arrhenius))
        arrow = "<=>" if reversible else "=>"
        rxn = {"equation": f"{text[0]} {arrow} {text[1]}"}
        if collider:
            rxn.update({"type": "falloff", "high-P-rate-constant": rate})
        elif thirdbody[0]:
            rxn.update({"type": "three-body", "rate-constant": rate})
        else:
            rxn["rate-constant"] = rate
        return rxn

    @staticmethod
    def _terms(side, names, equation):
        terms, hasm = [], False
        for token in (t.strip() for t in side.split("+")):
            if token == "M":
                hasm = True
            elif token in names:
                terms.append((token, 1.0))
            else:
                m = re.match(r"^(\d+\.?\d*|\.\d+)\s*(\S.*)$", token)
                if not m or m.group(2) not in names:
                    msg = f"{equation}: {token!r} is not a declared species"
                    raise ValueError(msg)
                terms.append((m.group(2), float(m.group(1))))
        return terms, hasm

    @classmethod
    def _auxline(cls, line, rxn, names):
        eq = rxn["equation"]
        rest = line
        for m in re.finditer(r"([^\s/]+)\s*/\s*([^/]*?)\s*/", line):
            key, values = m.group(1), m.group(2).split()
            kw = key.upper()
            rest = rest.replace(m.group(0), " ", 1)
            if kw == "LOW":
                rate = zip(("A", "b", "Ea"), map(cls._float, values))
                rxn["low-P-rate-constant"] = dict(rate)
            elif kw == "TROE":
                if len(values) not in (3, 4):
                    raise ValueError(f"{eq}: TROE takes three or four numbers")
                troe = zip(("A", "T3", "T1", "T2"), map(cls._float, values))
                rxn["Troe"] = dict(troe)
            elif kw in cls.refused:
                raise ValueError(f"{eq}: {kw} is not supported")
            elif key in names:
                rxn.setdefault("efficiencies", {})[key] = cls._float(values[0])
            else:
                msg = f"{eq}: {key!r} is neither a keyword nor a species"
                raise ValueError(msg)
        for word in rest.split():
            if word.upper() in ("DUP", "DUPLICATE"):
                rxn["duplicate"] = True
            else:
                raise ValueError(f"{eq}: cannot read {word!r}")
