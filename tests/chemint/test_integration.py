import numpy as np
import pytest

from tests.chemint.cases import at_cell, canterasdelay, cell, dQ, mechanism, oursdelay, stoichiometric, temperature


def test_one_substep_within_the_bound_is_the_finite_rate_source(backend):
    rng = np.random.default_rng(5)
    Y = rng.random(12)
    Y /= Y.sum()
    explicit = cell(backend, mechanism, "coupled", 10e5, 1800.0, Y)
    omega, rho = at_cell(explicit, "omega"), at_cell(explicit, "Q")[0]
    # the fastest species' way to its bound, in time
    headroom = np.where(omega < 0, Y, 1.0 - Y)
    dt = 0.5 * (rho * headroom / np.abs(omega)).min()
    explicit = cell(backend, mechanism, "coupled", 10e5, 1800.0, Y, dt=dt)
    substepped = cell(backend, mechanism, "sub-stepped", 10e5, 1800.0, Y, dt=dt, maxsubsteps=1)
    assert np.array_equal(dQ(explicit)[5:], dQ(substepped)[5:])


@pytest.mark.parametrize("maxsubsteps", [1, 10, 50])
def test_substeps_keep_the_state_in_bounds(backend, maxsubsteps):
    Y, dt = stoichiometric(), 1e-5
    explicit = cell(backend, mechanism, "coupled", 10e5, 2500.0, Y, dt=dt)
    Q, rhs = at_cell(explicit, "Q"), dQ(explicit)
    after = (Q[5:] + rhs[5:] * dt) / Q[0]
    assert after.min() < -0.1 or after.max() > 1.1
    substepped = cell(backend, mechanism, "sub-stepped", 10e5, 2500.0, Y, dt=dt, maxsubsteps=maxsubsteps)
    Q, rhs = at_cell(substepped, "Q"), dQ(substepped)
    after = (Q[5:] + rhs[5:] * dt) / Q[0]
    assert np.isfinite(rhs).all()
    assert after.min() > -1e-12 and after.sum() < 1.0 + 1e-12


def test_ignition_follows_cantera(backend):
    # measured 3.180e-5 s against Cantera's 3.175e-5 s at dt 1e-8; at dt 1e-7 the ten-substep cap throttles the source and ignition lands at 7.6e-5 s
    p0, T0 = 20e5, 1500.0
    theirs = canterasdelay(p0, T0, stoichiometric(), 400.0)
    dt = 1e-8
    mb = cell(backend, mechanism, "sub-stepped", p0, T0, stoichiometric(), dt=dt, maxsubsteps=10, equilibrium_tolerance=0.0)
    ours = oursdelay(mb, dt, T0, 400.0, 4 * theirs)
    assert ours is not None and abs(ours / theirs - 1.0) < 0.01
    # the exit never takes an igniting cell: the run with it is this one to the bit
    withexit = cell(backend, mechanism, "sub-stepped", p0, T0, stoichiometric(), dt=dt, maxsubsteps=10, equilibrium_tolerance=1e-7)
    assert oursdelay(withexit, dt, T0, 400.0, 4 * theirs) == ours
    assert np.array_equal(at_cell(withexit, "Q"), at_cell(mb, "Q"))


def test_single_precision_substeps_stay_finite_and_in_bounds(backend):
    # single stays finite and in [0, 1] but for a species at its bound overshooting by the precision's epsilon a substep; measured 6 K of 3930 K from double after three steps
    Y, dt = stoichiometric(), 1e-5
    runs = {}
    for precision in ("double", "single"):
        mb = cell(backend, mechanism, "sub-stepped", 10e5, 2500.0, Y, dt=dt, maxsubsteps=50, precision=precision)
        Q, rhs = at_cell(mb, "Q"), dQ(mb)
        assert np.isfinite(rhs).all()
        after = (Q[5:] + rhs[5:] * dt) / Q[0]
        slack = 50 * np.finfo(np.float32).eps
        assert after.min() > -slack and after.sum() < 1.0 + slack
        for _ in range(3):
            mb.step(dt)
        assert np.isfinite(at_cell(mb, "Q")).all()
        runs[precision] = temperature(mb)
    assert abs(runs["single"] / runs["double"] - 1.0) < 1e-2
