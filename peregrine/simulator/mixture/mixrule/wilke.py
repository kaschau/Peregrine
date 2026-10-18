import numpy as np

from peregrine.simulator.mixture.mixrule.base import BaseMixingRuleModel


class WilkeModel(BaseMixingRuleModel):
    name = "wilke"

    # phi_nm = (1 + s_n / s_m)^2 / sqrt(8 (1 + MW_n / MW_m)) with s = sqrt(mu) MW^(-1/4), so the pair loop is multiplies and adds
    def constants(self, species):
        MW = self.collect("MW", species)
        mwq = MW**-0.25
        pair = 1.0 / np.sqrt(8.0 * (1.0 + MW[:, None] / MW[None, :]))
        return dict(super().constants(species), MWqinv=mwq, wilkepair=pair)
