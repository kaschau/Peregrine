from pathlib import Path

import cantera as ct
import numpy as np
import pytest

from tests.eos.support import set_tguess, setstate, solver, state

mechanisms = str(Path(__file__).parent / "../../peregrine/simulator/mixture/database/mechanisms")


def fiterrors(gas, s):
    # e and h carry heats of formation and pass through zero, so they are judged against the sensible enthalpy cp T
    scale = gas.cp * gas.T
    return {
        "cp": abs(s["cp"] / gas.cp - 1.0),
        "gamma": abs(s["gamma"] / (gas.cp / gas.cv) - 1.0),
        "c": abs(s["c"] / gas.sound_speed - 1.0),
        "e": abs(s["e"] - gas.int_energy_mass) / scale,
        "h": abs(s["h"] - gas.enthalpy_mass) / scale,
    }


def closes(s, p, T, within):
    return abs(s["p"] / p - 1.0) < within and abs(s["T"] / T - 1.0) < within


@pytest.mark.parametrize("ctfile", ["C2H4_Air_Skeletal.yaml", "CH4_O2_FFCMY.yaml", "GRI30.yaml"])
def test_tpg(backend, ctfile):
    ct.add_directory(mechanisms)
    gas = ct.Solution(ctfile)
    rng = np.random.default_rng(5)
    p = rng.uniform(1e4, 1e5)
    T = rng.uniform(300.0, 1000.0)
    Y = rng.random(gas.n_species)
    Y /= Y.sum()
    gas.TPY = T, p, Y

    mb = solver(backend, ctfile, "tpg")
    assert mb.sim.mixture.eos.name == "tpg"
    # the bar is what the refit achieved for these species rather than refit-tol: the degree cap binds first
    achieved = max(sp["cpfiterr"] for sp in mb.sim.mixture.species.values()) + 1e-8

    setstate(mb, p, T, Y)
    s = state(mb)
    # rho pins Ru and every species' weight (measured 1e-15); p and T have already been through the Newton from the exact guess (measured 2.6e-12)
    assert abs(s["rho"] / gas.density - 1.0) < 1e-12
    assert closes(s, p, T, 1e-10)
    assert max(fiterrors(gas, s).values()) < achieved

    # the state again from the solution, the Newton started as far off as a stage moves it
    set_tguess(mb, by=0.01 * T)
    mb.postrhs()
    s = state(mb)
    assert closes(s, p, T, 1e-10)
    assert max(fiterrors(gas, s).values()) < achieved

    # and from the cold start
    set_tguess(mb, to=0.0)
    mb.postrhs()
    assert closes(state(mb), p, T, 1e-4)
