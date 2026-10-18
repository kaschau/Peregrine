import numpy as np
import pytest

import peregrine as pg
from peregrine.multiblock.interface import BaseInterface
from peregrine.plugins.solver import BaseSolverPlugin
from peregrine.simulator.mixture.constants import Ru

from tests.gases import air, primitives
from tests.integrators.cases import TurnedCubeMesher, box

# air's gas constant and ratio of specific heats, from the test gas's constants
R = Ru / air["Air"]["MW"]
gamma = air["Air"]["cp0"] / (air["Air"]["cp0"] - R)


class Landing(BaseSolverPlugin):
    name = "landing"

    def start(self, solver):
        self.acted = []

    def __call__(self, solver):
        self.acted.append((solver.nrt, solver.tme.get(), solver.dt.get()))


def flowing(mb, u, rot=None):
    prims = []
    for blk in mb.blocks:
        q = primitives(mb, blk)
        i = np.arange(q.shape[0])[:, None, None]
        q[..., 1:4] = u if rot is None else rot @ u
        # the wave is by cell index, not position, so a turned box carries the same values
        q[..., 4] = 300.0 * (1.0 + 0.1 * np.sin(2 * np.pi * i / 6.0))
        prims.append(q)
    mb.set_prims(prims)
    return prims


def closed_form(prims, mb, lengths, u):
    blk = mb.blocks[0]
    T = prims[0][blk.interior][..., 4]
    cmax = np.sqrt(gamma * R * T).max()
    d = np.asarray(lengths) / (np.asarray(blk.extents) - 1)
    acoustic = cmax * np.sqrt((1.0 / d**2).sum())
    convective = (np.abs(u) / d).sum()
    return dict(acoustic=acoustic, convective=convective, combined=acoustic + convective)


# a box of unequal cells, walled, and a velocity along every axis
lengths, dims, u = (0.6, 0.9, 1.5), (7, 6, 5), np.array([50.0, -30.0, 20.0])


def walled(backend, mesher=pg.mesher.CubeMesher, **kwargs):
    config, _ = box(controller="cfl", cfl=0.5, **kwargs)
    config["simulation"]["dt-max"] = 1.0
    config["bcvals"]["wall"] = {"kind": "slp-adia-wall"}
    mesh = mesher(mbdims=[1, 1, 1], blkdims=list(dims), lengths=list(lengths), defaultbc="wall")
    return config, pg.get_solver(config, mesh, backend(config))


def test_cflmax_is_the_fastest_oblique_mode(backend):
    config, mb = walled(backend)
    prims = flowing(mb, u)
    want = closed_form(prims, mb, lengths, u)
    got = mb.cflmax()
    for name in want:
        assert got[name] == pytest.approx(want[name], rel=1e-12)
    # the step is the target over the combined speed, under the ceiling
    assert mb.integ.controller.stepsize(mb) == config["simulation"]["cfl"] / got["combined"]
    config["simulation"]["dt-max"] = 1e-5
    assert mb.integ.controller.stepsize(mb) == 1e-5


def test_cflmax_at_rest_is_acoustic_alone(backend):
    config, mb = walled(backend)
    got = mb.cflmax()
    blk = mb.blocks[0]
    c = np.sqrt(gamma * R * 300.0)
    d = np.asarray(lengths) / (np.asarray(blk.extents) - 1)
    assert got["acoustic"] == pytest.approx(c * np.sqrt((1.0 / d**2).sum()), rel=1e-12)
    assert got["convective"] == 0.0 and got["combined"] == got["acoustic"]


def test_cflmax_is_invariant_to_turning_the_box(backend):
    # the box turned about a diagonal, the velocity with it: the normal speeds are the same numbers (measured within 6.3e-16)
    _, aligned = walled(backend)
    flowing(aligned, u)
    rot = BaseInterface.rotabout([1.0, 1.0, 1.0], 30.0)
    _, turned = walled(backend, mesher=lambda **kw: TurnedCubeMesher(rot, **kw))
    flowing(turned, u, rot)
    want, got = aligned.cflmax(), turned.cflmax()
    for name in want:
        assert got[name] == pytest.approx(want[name], rel=1e-12)


def test_the_step_lands_on_a_plugins_dt_out(backend):
    # a free step of about seven tenths of the plugin's interval: every second step is shortened to land on a multiple
    dtout = 1e-4
    config, mesh = box(controller="cfl", cfl=0.3, niter=8, mbdims=(2, 1, 1))
    config["simulation"]["dt-max"] = 1e-3
    config["ics"]["u"] = 10.0
    config["plugins"]["landing"] = {"dt-out": dtout}
    mb = pg.get_solver(config, mesh, backend(config))
    free = config["simulation"]["cfl"] / mb.cflmax()["combined"]
    assert 0.5 * dtout < free < dtout
    mb.run()
    acted = mb.plugins["landing"].acted
    assert [nrt for nrt, _, _ in acted] == [2, 4, 6, 8]
    for n, (nrt, tme, dt) in enumerate(acted, start=1):
        assert tme == pytest.approx(n * dtout, rel=1e-12)
        # the landing step is what the free step left of the interval
        assert dt == pytest.approx(n * dtout - (n - 1) * dtout - free, rel=1e-12)
    assert mb.tme.get() == pytest.approx(4 * dtout, rel=1e-12)


@pytest.mark.parametrize(
    "cfl, bounded",
    [(1.6, True), (1.7, True), (1.8, False), (2.0, False)],
)
def test_the_target_brackets_the_schemes_limit(backend, cfl, bounded):
    # a line of forty cells, tvd-rk3 over the two-point central flux, the four-cell mode seeded in the pressure over a temperature wave at rest: it decays under sqrt(3), the scheme's reach along the imaginary axis, and grows past it
    dx, seed = 1e-3, 1e-6
    config, mesh = box(
        controller="cfl",
        cfl=cfl,
        niter=300,
        dims=(41, 2, 2),
        lengths=(40 * dx, dx, dx),
        periodic=(True, False, False),
        names={n: "slip" for n in (3, 4, 5, 6)},
    )
    config["simulation"]["dt-max"] = 1.0
    config["bcvals"]["slip"] = {"kind": "slp-adia-wall"}
    mb = pg.get_solver(config, mesh, backend(config))
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    x = blk.xcc.get()[..., 0]
    q[..., 0] = 101325.0 * (1.0 + seed * np.sin(2 * np.pi * x / (4 * dx)))
    q[..., 4] = 300.0 * (1.0 + 0.1 * np.sin(2 * np.pi * x / (40 * dx)))
    mb.set_prims([q])
    assert mb.integ.controller.stepsize(mb) == cfl / mb.cflmax()["combined"]
    mb.run()
    p = blk.state.get()[blk.interior][..., list(mb.sim.statefields).index("p")]
    scatter = np.abs(p / 101325.0 - 1.0).max()
    assert np.isfinite(scatter)
    # measured 6.6e-13 and 1.0e-9 under the limit, 5.6e-5 and 0.13 over it
    if bounded:
        assert scatter < seed
    else:
        assert scatter > 10 * seed
