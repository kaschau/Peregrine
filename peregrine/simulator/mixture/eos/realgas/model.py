import numpy as np

from peregrine.codegen import Struct
from peregrine.simulator.mixture.constants import Ru, kB
from peregrine.simulator.mixture.caloric.tdepcp.model import TempDepCpModel
from peregrine.simulator.mixture.eos.base import BaseEosModel
from peregrine.simulator.mixture.eos.tpg.model import TpgModel


class RealGasModel(BaseEosModel):
    name = "real-gas"
    critical = ("Tcrit", "pcrit", "Vcrit", "acentric")
    # the thermally perfect moments plus the cubic's sums that do not move with T, b, c and S2, each over Y_n / MW_n and scaled by MWmix in the kernel
    moments = dict(TpgModel.moments)
    moments.update(b="cubicbMW", c="vshiftMW", da="cubicdaMW")
    # the cubic at a point, passed between its macros; and what every species' partial enthalpy shares there beyond it
    def structs(self, species):
        cubic = Struct("Cubic", (
            "MWmix", "bm", "cm", "S1", "S2", "am", "dam",
            "Astar", "Bstar", "Cstar", "invRuT", "Z", "z1", "z2", "invslope", "dZdT", "v", "I", "logratio", "q", "qd",
        ))
        hpoint = Struct("Hpoint", (
            "lnT", "Tinv", "sqrtT", "Iz", "IzT", "dlnZmB", "invS1", "invBm", "invamsqrtT",
        ))
        return (cubic, hpoint)
    # Soave-Redlich-Kwong: p = Ru T / (v - b) - a / (v^2 + u b v + w b^2), alpha = (1 + m (1 - sqrt(T / Tc)))^2 with m = m0 + m1 w + m2 w^2
    cubic = dict(
        u=1.0, w=0.0, omega_a=0.4274802, omega_b=0.0866403,
        m=(0.480, 1.574, -0.176),
    )
    def computes(self, ns):
        """The base's, and the cubic's root Z at the point, which the species
        enthalpies at a stored state read in place of a solve."""
        return super().computes(ns) + ("Z",)

    def __init__(self, cfgsect):
        super().__init__(cfgsect, TempDepCpModel)
        self.needs += self.critical
        self.derivable[self.critical] = ("well", "diam")
        # the point, which the cubic's constants and a dense transport read
        self.provides["critpoint"] = self.critical

    def constants(self, species):
        c = self.cubic
        # the roots of the form's quadratic, v^2 + u b v + w b^2 = (v + sigma b)(v + eps b), which the residual terms are written in
        sq = np.sqrt(c["u"] ** 2 - 4.0 * c["w"])
        vshift = [sp.get("vshift", 0.0) for sp in species.values()]
        constants = {
            "cubicu": np.array(c["u"]),
            "cubicw": np.array(c["w"]),
            "cubicsigma": np.array(0.5 * (c["u"] + sq)),
            "cubiceps": np.array(0.5 * (c["u"] - sq)),
            "vshift": np.array(vshift),
        }
        for key in self.critical:
            constants[key] = self.collect(key, species)
        keys = ("cubicsqrtac", "cubicb", "cubicda", "cubicmTc", "cubiconepm")
        constants.update(zip(keys, self.cubicconsts(species)))
        # the mole-fraction sums' constants, per unit of Y_n / MW_n
        MWinv = 1.0 / self.collect("MW", species)
        constants["cubicsqrtacMW"] = constants["cubicsqrtac"] * MWinv
        constants["cubicbMW"] = constants["cubicb"] * MWinv
        constants["vshiftMW"] = constants["vshift"] * MWinv
        constants["cubicdaMW"] = constants["cubicda"] * MWinv
        return dict(super().constants(species), **constants)

    # a species without a critical point gets one from its Lennard-Jones parameters: Chung's relations (transport/chung.py) run backwards, and Pitzer's Zc at w = 0
    def critpoint(self, species):
        out = {key: [] for key in self.critical}
        for sp in species.values():
            if sp.missing(self.critical):
                Tc = 1.2593 * sp["well"] / kB
                # cm3/mol -> m3/kmol
                Vcmol = (sp["diam"] * 1e10 / 0.809) ** 3 * 1e-3
                got = {
                    "Tcrit": Tc,
                    # Zc = 0.291 - 0.08 w at w = 0
                    "pcrit": 0.291 * Ru * Tc / Vcmol,
                    "Vcrit": Vcmol / sp["MW"],
                    "acentric": 0.0,
                }
            else:
                got = {key: sp[key] for key in self.critical}
            for key in self.critical:
                out[key].append(got[key])
        return tuple(out[key] for key in self.critical)

    def cubicconsts(self, species):
        Tc = self.collect("Tcrit", species)
        pc = self.collect("pcrit", species)
        w = self.collect("acentric", species)
        c = self.cubic
        sqrtac = np.sqrt(c["omega_a"] / pc) * Ru * Tc
        b = c["omega_b"] * Ru * Tc / pc
        m = c["m"][0] + c["m"][1] * w + c["m"][2] * w**2
        invsqrtTc = Tc**-0.5
        # sqrt(ac) m / sqrt(Tc) is -(da/dT) / sqrt(a) times sqrt(T); m / sqrt(Tc) and 1 + m open Soave's modulus |1 + m (1 - sqrt(T / Tc))| as |(1 + m) - sqrt(T) m / sqrt(Tc)|
        return sqrtac, b, sqrtac * m * invsqrtTc, m * invsqrtTc, 1.0 + m
