from peregrine.simulator.mixture.caloric.tdepcp.model import TempDepCpModel
from peregrine.simulator.mixture.eos.base import BaseEosModel


class TpgModel(BaseEosModel):
    name = "tpg"
    # the gas constant, cp in ln T and h's reference premultiplied by the species' gas constants, so the mixture's cp is one Horner and h's coefficients follow from its row at the point
    moments = dict(R="Rsp", cpcoeff="cppolyR", h298="h298R")

    def __init__(self, cfgsect):
        super().__init__(cfgsect, TempDepCpModel)
