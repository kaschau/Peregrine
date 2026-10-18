import pytest

import peregrine as pg
from peregrine.config import ConfigError

from tests.gases import configure


def simulation(physics, gas="air"):
    config = pg.Config()
    configure(config, gas, physics)
    return config, pg.simulator.get_simulator(config)


def test_euler_declares_its_vectors():
    _, sim = simulation("euler")
    assert sim.iovars.privars == ["p", "u", "v", "w", "T"]
    assert sim.iovars.exported == ["rho", "p", "u", "v", "w", "T", "Air"]
    assert sim.convars == ["rho", "rhou", "rhov", "rhow", "rhoE"]
    assert sim.ne == 5 and sim.arrays()["Q"]["comps"] == 5


def test_species_widen_the_primitive_vector():
    _, sim = simulation("navier-stokes", "CH4_O2")
    names = sim.mixture.spnames
    assert len(names) > 1
    # the last species is what the others leave: it is exported, never carried
    assert sim.iovars.privars == ["p", "u", "v", "w", "T"] + names[:-1]
    assert sim.iovars.exported == ["rho", "p", "u", "v", "w", "T"] + names
    assert sim.convars[5:] == [f"rho{s}" for s in names[:-1]]
    assert sim.ne == 5 + len(names) - 1 and sim.arrays()["Q"]["comps"] == sim.ne


def test_the_initial_state_names_species_by_name():
    config, sim = simulation("euler", "CH4_O2")
    config["ics"]["Y"] = {"CH4": 0.2, "O2": 0.8}
    state = sim.ics()
    names = sim.mixture.spnames
    assert len(state) == sim.ne
    assert state[:5] == [config["ics"][k] for k in ("p", "u", "v", "w", "T")]
    assert state[5 + names.index("CH4")] == 0.2 and state[5 + names.index("O2")] == 0.8
    # every other carried species is none; the last takes the remainder, so it has no slot
    others = [state[5 + n] for n, s in enumerate(names[:-1]) if s not in ("CH4", "O2")]
    assert all(y == 0.0 for y in others)
    config["ics"]["Y"] = {"Xe": 1.0}
    with pytest.raises(ConfigError):
        sim.ics()
