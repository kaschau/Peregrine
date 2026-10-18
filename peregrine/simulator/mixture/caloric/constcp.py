import numpy as np

from peregrine.simulator.mixture.constants import Ru, Tref
from peregrine.simulator.mixture.caloric.base import BaseCaloricModel


class ConstantCpModel(BaseCaloricModel):
    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        self.needs += ("cp0",)

    def cppoly(self, sp):
        # a constant is its own fit, exactly
        return [sp["cp0"] * sp["MW"] / Ru], 0.0

    def hpoly(self, sp):
        return sp["cppoly"]

    def h298(self, sp):
        # J/kmol; a non-reacting case need not state it
        return sp.get("dHf298", 0.0) / Ru - Tref * sp["cppoly"][0]

    def spoly(self, sp):
        # J/kmol/K; a non-reacting case need not state it
        c = sp["cppoly"][0]
        return [sp.get("s298", 0.0) / Ru - c * np.log(Tref), c]
