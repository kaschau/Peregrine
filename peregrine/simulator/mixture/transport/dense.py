from peregrine.simulator.mixture.eos.realgas.model import RealGasModel
from peregrine.simulator.mixture.transport.kinetic import KineticTheoryModel


class DenseGasModel(KineticTheoryModel):
    # a base of the dense models, which the config names by their own names
    name = None

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        # the same need as the real gas eos, which will have put the point on the species
        self.derivable[RealGasModel.critical] = ("well", "diam")
        self.deps = (RealGasModel,)
