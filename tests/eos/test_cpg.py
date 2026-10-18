from pathlib import Path

import cantera as ct
import numpy as np

from tests.eos.support import set_tguess, setstate, solver, state

ctfile = str(Path(__file__).parent / "ct_test_cpg.yaml")


def relative(reference, got):
    return {name: abs(got[name] / reference[name] - 1.0) for name in reference}


def test_cpg(backend):
    gas = ct.Solution(ctfile)
    rng = np.random.default_rng(2)
    p = rng.uniform(1e4, 1e5)
    T = rng.uniform(100.0, 1000.0)
    Y = rng.random(gas.n_species)
    Y /= Y.sum()
    gas.TPY = T, p, Y

    mb = solver(backend, ctfile, "cpg")
    assert mb.sim.mixture.eos.name == "cpg"
    setstate(mb, p, T, Y)
    reference = dict(
        rho=gas.density,
        p=p,
        T=T,
        e=gas.int_energy_mass,
        h=gas.enthalpy_mass,
        cp=gas.cp,
        gamma=gas.cp / gas.cv,
        c=gas.sound_speed,
    )
    assert max(relative(reference, state(mb)).values()) < 1e-10

    # the state again from the solution, the temperature scrambled first
    set_tguess(mb, to=0.0)
    mb.postrhs()
    assert max(relative(reference, state(mb)).values()) < 1e-10
