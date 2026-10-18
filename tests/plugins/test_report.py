import re

import numpy as np
import pytest

import peregrine as pg
from peregrine.simulator.mixture import Ru

from tests.gases import air, configure, primitives

# a 1 mm cube of cpg air between slip walls; blkdims count nodes, so 5 is 4 cells of 2.5e-4
L = 1e-3
U = (10.0, 20.0, 30.0)


def box(backend, blkdims=(5, 5, 5), **simulation):
    config = pg.Config()
    configure(config, "air", "euler")
    config["simulation"].update({"controller": "fixed", "dt": 1e-6, **simulation})
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    config["plugins"]["report"] = {"niter-out": 1000}
    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1],
        blkdims=list(blkdims),
        lengths=[L] * 3,
        bcnames=dict.fromkeys(range(1, 7), "walls"),
    )
    return pg.get_solver(config, mesh, backend(config))


def sound_speed(T=300.0):
    R = Ru / air["Air"]["MW"]
    gamma = air["Air"]["cp0"] / (air["Air"]["cp0"] - R)
    return np.sqrt(gamma * R * T)


def uniform(mb, u):
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    q[..., 1:4] = u
    mb.set_prims([q])


def printed(out, name):
    return float(re.search(rf"{name}\s*:\s*(\S+)", out).group(1))


def test_the_speeds_are_the_analytic_ones_on_a_uniform_state(backend):
    mb = box(backend)
    uniform(mb, U)
    dx = L / 4
    c = sound_speed()
    blk = mb.blocks[0]
    state = blk.state.get()[blk.interior]
    assert np.allclose(state[..., mb.sim.statefields.index("c")], c, rtol=1e-12)
    # an isotropic cell's acoustic length is dx / sqrt(3); the convective speed sums the axes' normal speeds
    acoustic, convective = c * np.sqrt(3.0) / dx, sum(U) / dx
    cfl = mb.cflmax()
    assert cfl["acoustic"] == pytest.approx(acoustic, rel=1e-12)
    assert cfl["convective"] == pytest.approx(convective, rel=1e-12)
    assert cfl["combined"] == pytest.approx(acoustic + convective, rel=1e-12)


def test_an_axis_one_cell_thick_is_infinitely_long(backend):
    mb = box(backend, blkdims=(5, 2, 2))
    uniform(mb, U)
    dx = L / 4
    c = sound_speed()
    cfl = mb.cflmax()
    assert cfl["acoustic"] == pytest.approx(c / dx, rel=1e-12)
    assert cfl["convective"] == pytest.approx(U[0] / dx, rel=1e-12)
    assert cfl["combined"] == pytest.approx((c + U[0]) / dx, rel=1e-12)


def test_the_cfl_controller_divides_by_the_combined_speed(backend):
    mb = box(backend, controller="cfl", cfl=0.5, **{"dt-max": 1.0})
    uniform(mb, U)
    dx = L / 4
    combined = (sound_speed() * np.sqrt(3.0) + sum(U)) / dx
    assert mb.integ.controller.stepsize(mb) == pytest.approx(0.5 / combined, rel=1e-12)


def test_the_step_line_prints_the_cfl_of_the_step(backend, capsys):
    mb = box(backend)
    report = mb.plugins["report"]
    uniform(mb, U)
    dt = 1e-6
    mb.dt.set(dt)
    capsys.readouterr()
    report(mb)
    out = capsys.readouterr().out
    dx = L / 4
    acoustic, convective = sound_speed() * np.sqrt(3.0) / dx * dt, sum(U) / dx * dt
    # printed to three decimals, 2.406, 0.240 and 2.646, so each is within half a thousandth
    assert printed(out, "acoustic") == pytest.approx(acoustic, abs=5e-4)
    assert printed(out, "convective") == pytest.approx(convective, abs=5e-4)
    assert printed(out, "MAX CFL") == pytest.approx(acoustic + convective, abs=5e-4)


def test_the_closing_line_counts_this_runs_steps(backend, capsys):
    mb = box(backend)
    report = mb.plugins["report"]
    # a restart: the case has taken a thousand steps before this run starts
    mb.nrt = 1000
    report.start(mb)
    capsys.readouterr()
    for _ in range(3):
        mb.step(1e-6)
    assert report.nsteps(mb) == 3
    report.finalize(mb)
    out = capsys.readouterr().out
    assert "over 3 steps" in out
