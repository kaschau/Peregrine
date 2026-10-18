import numpy as np

from peregrine.simulator.mixture.mixrule.base import BaseMixingRuleModel


# within a few percent of Wilke for like molecules; the light species (H2, H) are where it strays
class HerningModel(BaseMixingRuleModel):
    name = "herning"

    def constants(self, species):
        MW = self.collect("MW", species)
        return dict(super().constants(species), sqrtMW=np.sqrt(MW))
