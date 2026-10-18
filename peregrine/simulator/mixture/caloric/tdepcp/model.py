import numpy as np

from peregrine.simulator.mixture.constants import Tref
from peregrine.simulator.mixture.species import AnyOf
from peregrine.misc import subclasses
from peregrine.simulator.mixture.polyfit import PolyFitMixin
from peregrine.simulator.mixture.species import Species
from peregrine.simulator.mixture.thermo import BaseThermoData
from peregrine.simulator.mixture.caloric.base import BaseCaloricModel


class TempDepCpModel(BaseCaloricModel, PolyFitMixin):
    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        self.needs += (AnyOf(f.name for f in subclasses(BaseThermoData)),)

    def cppoly(self, sp):
        T, cpR = self._sample(sp)
        tol, deg = self.cfgsect["refit-tol"], self.cfgsect["refit-max-deg"]
        return self.fit(np.log(T), cpR, tol, deg)

    def hpoly(self, sp):
        return self.expintegral(sp["cppoly"])

    def h298(self, sp):
        h298R, _ = BaseThermoData.find(sp).hsref_r(sp)
        return h298R - Tref * self.evaluate(sp["hpoly"], np.log(Tref))

    def spoly(self, sp):
        _, srefR = BaseThermoData.find(sp).hsref_r(sp)
        S = self.integral(sp["cppoly"])
        S[0] = srefR - self.evaluate(S, np.log(Tref))
        return S

    def _sample(self, sp):
        Tlow, Thigh = self.cfgsect["T-range"]
        T, cpR = BaseThermoData.find(sp).cp_r(sp)
        if Tlow < T[0]:
            T, cpR = self._pad_low(sp, T, cpR, Tlow)
        if Thigh > T[-1]:
            T, cpR = self._pad_high(T, cpR, Thigh)
        inside = (T >= Tlow) & (T <= Thigh)
        return T[inside], cpR[inside]

    def _pad_low(self, sp, T, cpR, Tlow):
        # cp/R at 0 K: translation, and rot if the shape is known
        known = "geometry" in sp
        frozen = 2.5 + Species.rotdof[sp["geometry"]] if known else cpR[0]
        dT = T[1] - T[0]
        Tx = np.arange(T[0] - dT, Tlow, -dT)
        if Tx.size == 0 or Tx[-1] > Tlow:
            Tx = np.append(Tx, Tlow)
        Tx = Tx[::-1]
        ramp = frozen + (cpR[0] - frozen) * Tx / T[0]
        return np.concatenate([Tx, T]), np.concatenate([ramp, cpR])

    @staticmethod
    def _pad_high(T, cpR, Thigh):
        dT = T[-1] - T[-2]
        Tx = np.arange(T[-1] + dT, Thigh, dT)
        if Tx.size == 0 or Tx[-1] < Thigh:
            Tx = np.append(Tx, Thigh)
        held = np.full_like(Tx, cpR[-1])
        return np.concatenate([T, Tx]), np.concatenate([cpR, held])
