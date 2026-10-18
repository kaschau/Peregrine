from peregrine.simulator.mixture.transport.dense import DenseGasModel


# resmu and reskappa are fitted on Soave-Redlich-Kwong; another cubic needs its own
class EntropyScalingModel(DenseGasModel):
    name = "entropy-scaling"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        self.needs += ("resmu", "reskappa")

    def constants(self, species):
        return dict(
            super().constants(species),
            resmu=self.collect_poly("resmu", species),
            reskappa=self.collect_poly("reskappa", species),
            rsqrtMW=self.collect("MW", species) ** -0.5,
        )
