from pathlib import Path

import cantera as ct
import numpy as np
import pytest

import peregrine as pg
from peregrine.simulator.mixture import Mixture

mechanisms = ("GRI30.yaml", "CH4_O2_FFCMY.yaml", "C2H4_Air_Skeletal.yaml")
# none at the NASA7 switch, where Cantera's two ranges disagree with each other
temperatures = (400.0, 700.0, 950.0, 1200.0, 1800.0, 2500.0, 3200.0)


def mechpath(name):
    import peregrine.simulator.mixture as mix

    return str(Path(mix.__file__).parent / "database" / "mechanisms" / name)


@pytest.fixture(scope="module", params=mechanisms)
def mech(request):
    name = request.param
    sect = dict(pg.Config()["mixture"])
    sect.update({"species": name, "eos": "tpg", "T-range": (300.0, 3500.0)})
    mixture = Mixture(sect)
    return mixture, mixture.eos.constants(mixture.species), ct.Solution(mechpath(name))


def horner(rows, u):
    return np.array([np.polyval(np.flip(row), u) for row in rows])


def test_the_fits_are_canteras(mech):
    mixture, constants, gas = mech
    achieved = max(sp["cpfiterr"] for sp in mixture.species.values()) + 1e-8
    for T in temperatures:
        gas.TP = T, ct.one_atm
        # past a species' own range the fit holds cp and Cantera extrapolates
        inside = np.array([gas.species(i).thermo.max_temp >= T for i in range(gas.n_species)])
        assert inside.sum() >= gas.n_species - 3
        u = np.log(T)
        cp = horner(constants["cppoly"], u)
        h = horner(constants["hpoly"], u) + constants["h298"] / T
        s = horner(constants["spoly"], u)
        g = mixture.eos.gibbs(mixture.species, [T])[:, 0]
        assert np.abs(cp / gas.standard_cp_R - 1.0)[inside].max() < achieved
        assert (np.abs(h - gas.standard_enthalpies_RT) / gas.standard_cp_R)[inside].max() < achieved
        assert np.abs(s / gas.standard_entropies_R - 1.0)[inside].max() < achieved
        # g / (Ru T), what the reverse rates exponentiate, holds to 2e-3 absolute as the refit's integral
        assert np.abs(g - gas.standard_gibbs_RT)[inside].max() < 2e-3
