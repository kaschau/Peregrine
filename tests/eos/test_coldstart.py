import numpy as np
import pytest

import peregrine as pg

from tests.eos.support import config, set_tguess, mesh, setstate, solver, state

FLOOR = 1e-6


# the guesses sit a thousand kelvin off, where a restart's padded halo across a front leaves a cell's last temperature
@pytest.mark.parametrize("to", [3400.0, 320.0])
def test_the_thermally_perfect_gas_converges_from_a_cold_start(backend, to):
    mb = solver(backend, ["N2"], "tpg")
    setstate(mb, 1e5, 1200.0, [1.0])
    set_tguess(mb, to=to)
    mb.postrhs()
    assert state(mb)["T"] == pytest.approx(1200.0, abs=FLOOR)


@pytest.mark.parametrize("to", [1500.0, 110.0])
def test_the_real_gas_converges_from_a_cold_start(backend, to):
    # dense methane: 6 bar per kelvin at fixed density, where three steps leave kelvins
    mb = solver(backend, ["CH4"], "real-gas", trange=(100.0, 3500.0))
    setstate(mb, 3e7, 200.0, [1.0])
    set_tguess(mb, to=to)
    mb.postrhs()
    assert state(mb)["T"] == pytest.approx(200.0, abs=FLOOR)


@pytest.mark.parametrize("to", [1500.0, 110.0])
def test_the_double_flux_converges_from_a_cold_start(backend, to):
    cfg = config(["CH4"], "real-gas-df", trange=(100.0, 3500.0))
    cfg["simulation"]["simulator"] = "euler-df"
    mb = pg.get_solver(cfg, mesh(), backend(cfg))
    setstate(mb, 3e7, 200.0, [1.0])
    set_tguess(mb, to=to)
    mb.postrhs()
    assert state(mb)["T"] == pytest.approx(200.0, abs=FLOOR)


def test_a_warm_start_is_the_same_state(backend):
    # the state from the converged guess, and from the guess a stage's drift away: the same three steps
    mb = solver(backend, ["CH4"], "real-gas", trange=(100.0, 3500.0))
    setstate(mb, 3e7, 200.0, [1.0])
    mb.postrhs()
    warm = dict(state(mb))
    set_tguess(mb, by=0.5)
    mb.postrhs()
    moved = state(mb)
    assert np.allclose([moved[k] for k in warm], [warm[k] for k in warm], rtol=1e-10, atol=0.0)
