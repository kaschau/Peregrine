from peregrine.simulator.mixture.transport.base import BaseTransportModel


class ConstantPropsModel(BaseTransportModel):
    name = "constant"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        self.fromcase += ("mu0", "kappa0")

    def constants(self, species):
        mu0 = self.collect("mu0", species)
        kappa0 = self.collect("kappa0", species)
        return dict(super().constants(species), mu0=mu0, kappa0=kappa0)
