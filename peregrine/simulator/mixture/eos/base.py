import numpy as np

from peregrine.simulator.mixture.constants import Ru
from peregrine.simulator.mixture.base import BaseModel


class BaseEosModel(BaseModel):
    def __init__(self, cfgsect, calmodel):
        super().__init__(cfgsect)
        self.caloric = calmodel(cfgsect)
        self.needs += self.caloric.needs
        # ordered: each reads what the one before stored
        self.provides["cppoly"] = ("cppoly", "cpfiterr")
        self.provides.update(hpoly="hpoly", h298="h298", spoly="spoly")

    @classmethod
    def template(cls):
        return f"{cls.__module__.rsplit('.', 1)[0]}.kernels.eos"

    def cppoly(self, species):
        fits = [self.caloric.cppoly(sp) for sp in species.values()]
        return [fit for fit, _ in fits], [error for _, error in fits]

    def hpoly(self, species):
        return [self.caloric.hpoly(sp) for sp in species.values()]

    def h298(self, species):
        return [self.caloric.h298(sp) for sp in species.values()]

    def spoly(self, species):
        return [self.caloric.spoly(sp) for sp in species.values()]

    def constants(self, species):
        cppoly = self.collect_poly("cppoly", species)
        hpoly = self.collect_poly("hpoly", species)
        h298 = self.collect("h298", species)
        spoly = self.collect_poly("spoly", species)
        Rsp = Ru / self.collect("MW", species)
        return dict(
            super().constants(species),
            cppoly=cppoly,
            hpoly=hpoly,
            h298=h298,
            spoly=spoly,
            gpoly=self.gpoly(species),
            cppolyR=cppoly * Rsp[:, None],
            hpolyR=hpoly * Rsp[:, None],
            h298R=h298 * Rsp,
        )

    # g / (Ru T) less h298 / T: the enthalpy's polynomial in ln T less the entropy's
    @classmethod
    def gpoly(cls, species):
        hpoly, spoly = cls.collect_poly("hpoly", species), cls.collect_poly("spoly", species)
        gpoly = np.zeros((len(species), max(hpoly.shape[1], spoly.shape[1])))
        gpoly[:, : hpoly.shape[1]] += hpoly
        gpoly[:, : spoly.shape[1]] -= spoly
        return gpoly

    # u is ln T, given when it was taken another way than the library's
    @classmethod
    def gibbs(cls, species, T, u=None):
        u = np.log(T) if u is None else np.asarray(u, dtype=np.float64)
        Tinv = 1.0 / np.asarray(T, dtype=np.float64)
        gpoly, h298 = cls.gpoly(species), cls.collect("h298", species)
        poly = [np.polyval(np.flip(g), u) for g in gpoly]
        return np.array([p + h * Tinv for p, h in zip(poly, h298)])

    # the members of EosState; rhoinv is 1 / rho as stored, the bits a kernel's division from Q[0] would give; h = e + p / rho and cv = cp / gamma are formed where they are read
    def computes(self, ns):
        return ("p", "T", "rhoinv", "rho", "e", "cp", "gamma", "c")

    # member -> the species constant the mixing macros sum over the composition once per point, for this eos's algebra to run on
    moments = {}
