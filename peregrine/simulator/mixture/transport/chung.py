import numpy as np

from peregrine.simulator.mixture.species import database
from peregrine.simulator.mixture.transport.dense import DenseGasModel


# Chung's dipole terms are not applied; the polar physics is the collision integrals'
class ChungDenseGasModel(DenseGasModel):
    name = "chung"

    # the acentric factor the kernel reads is the real gas's constant
    def constants(self, species):
        chung = database("chung")
        sigma, epsk = self.chungparams(species)
        return dict(
            super().constants(species),
            chungsigma=sigma,
            chungepsk=epsk,
            chungvisc=np.array(chung["viscosity"], dtype=np.float64),
            chungcond=np.array(chung["conductivity"], dtype=np.float64),
        )

    # Chung's own units, which the correlation takes: sigma in angstrom from Vc in cm3/mol, epsilon/k in kelvin
    def chungparams(self, species):
        MW = self.collect("MW", species)
        Vc = self.collect("Vcrit", species) * MW * 1e3
        sigma = 0.809 * Vc ** (1.0 / 3.0)
        epsk = self.collect("Tcrit", species) / 1.2593
        return sigma, epsk
