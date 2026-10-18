from peregrine.simulator.mixture.caloric.constcp import ConstantCpModel
from peregrine.simulator.mixture.eos.base import BaseEosModel


class CpgModel(BaseEosModel):
    name = "cpg"
    # the mixture's gas constant, cp and reference enthalpy are all a perfect gas's algebra needs
    moments = {"R": "Rsp", "cp": "cp0", "h298": "h298R"}

    def __init__(self, cfgsect):
        super().__init__(cfgsect, ConstantCpModel)

    def constants(self, species):
        cp0 = self.collect("cp0", species)
        return dict(super().constants(species), cp0=cp0)
