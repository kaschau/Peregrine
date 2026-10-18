import numpy as np

from peregrine.simulator.mixture.diffusion.base import BaseSpeciesDiffusionModel


class LewisModel(BaseSpeciesDiffusionModel):
    name = "lewis"

    def lewis(self, species):
        return np.array([sp.get("lewis", 1.0) for sp in species.values()])

    def constants(self, species):
        return dict(super().constants(species), lewis=self.lewis(species))
