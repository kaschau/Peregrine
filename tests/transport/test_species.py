import numpy as np
import pytest
from peregrine.simulator.mixture import Mixture

from tests.transport import cases


@pytest.mark.parametrize("ctfile", cases.mechanisms)
def test_species_fits(ctfile):
    gas = cases.cantera(ctfile)
    config = cases.mixture_config(
        ctfile, eos="tpg", trans="kinetic-theory", diffusion="binary", **{"T-range": cases.trange}
    )
    mixture = Mixture(config["mixture"])
    assert mixture.spnames == gas.species_names
    ns, p = mixture.ns, 101325.0
    constants = mixture.constants()
    worst = {"mu": 0.0, "D": 0.0, "kappa": 0.0, "kappa interior": 0.0}
    for T in np.linspace(*cases.trange, 33):
        mu, kappa = cases.species_fits(constants, T)
        D = cases.pair_fits(constants, T)
        gas.TPX = T, p, np.ones(ns) / ns
        worst["mu"] = max(worst["mu"], np.abs(mu / gas.species_viscosities - 1.0).max())
        worst["D"] = max(worst["D"], np.abs(D / p / gas.binary_diff_coeffs - 1.0).max())
        # pure, every mixing rule gives the species' own conductivity
        for n in range(ns):
            Y = np.zeros(ns)
            Y[n] = 1.0
            gas.TPY = T, p, Y
            off = abs(kappa[n] / gas.thermal_conductivity - 1.0)
            worst["kappa"] = max(worst["kappa"], off)
            if 800.0 <= T <= 3000.0:
                worst["kappa interior"] = max(worst["kappa interior"], off)
    print(ctfile, {k: f"{v:.2e}" for k, v in worst.items()})
    # measured 0.65 % and 0.67 %: the refit tolerance on both sides
    assert worst["mu"] < 1e-2
    assert worst["D"] < 1e-2
    # Cantera fits its conductivity at a fixed degree over the species' thermo range, loose at that range's ends; between 800 and 3000 K the worst is 1.07 %
    assert worst["kappa interior"] < 1.5e-2
    assert worst["kappa"] < 4e-2
