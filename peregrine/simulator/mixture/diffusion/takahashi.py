from peregrine.simulator.mixture.eos.realgas.model import RealGasModel
from peregrine.simulator.mixture.diffusion.binary import BinaryModel


class TakahashiModel(BinaryModel):
    name = "takahashi"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        # the critical point the real gas eos puts on the species
        self.derivable[RealGasModel.critical] = ("well", "diam")
        # the critical point the kernel reduces by is the real gas's constant
        self.deps += (RealGasModel,)
