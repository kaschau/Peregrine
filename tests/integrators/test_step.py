import numpy as np
import pytest

import peregrine as pg
from peregrine.plugins.solver import BaseSolverPlugin
from peregrine.simulator.mixture.constants import Ru

from tests.gases import air, primitives
from tests.integrators.cases import box, perturb, schemes, totals


class Counter(BaseSolverPlugin):
    name = "counter"

    def start(self, solver):
        self.acted, self.finalized = [], 0

    def __call__(self, solver):
        self.acted.append((solver.nrt, solver.tme.get()))

    def finalize(self, solver):
        self.finalized += 1


def uniform(physics, integrator, controller, **kwargs):
    config, mesh = box(physics=physics, integrator=integrator, controller=controller, **kwargs)
    config["ics"]["u"] = 10.0
    return config, mesh


def assert_unchanged(mb, before):
    blk = mb.blocks[0]
    after = blk.Q.get()
    assert np.isfinite(after).all()
    # against the largest component; the transverse momenta are zero
    scale = np.abs(before[blk.interior]).max()
    assert np.abs(after[blk.interior] - before[blk.interior]).max() < 1e-10 * scale


@pytest.mark.parametrize("integrator", schemes)
@pytest.mark.parametrize("physics", ["euler", "navier-stokes"])
@pytest.mark.parametrize("controller", ["fixed", "cfl"])
def test_step(backend, integrator, physics, controller):
    config, mesh = uniform(physics, integrator, controller, niter=3)
    mb = pg.get_solver(config, mesh, backend(config))
    assert type(mb.integ).name == integrator
    assert type(mb.integ.controller).name == controller
    assert type(mb.sim).name == physics
    dt = mb.integ.controller.stepsize(mb)
    if controller == "fixed":
        assert dt == config["simulation"]["dt"]
    else:
        assert dt == min(config["simulation"]["cfl"] / mb.cflmax()["combined"], config["simulation"]["dt-max"])
    Q0 = mb.blocks[0].Q.get()
    mb.run()
    assert_unchanged(mb, Q0)
    assert mb.nrt == 3 and mb.tme.get() == pytest.approx(3 * dt, rel=1e-15)


def test_run_takes_the_steps_the_controller_sizes(backend):
    # a perturbed box, so the steps change the state: the loop's three steps are three steps by hand, bitwise
    config, mesh = box(physics="navier-stokes", dt=2e-6, niter=3)
    config["plugins"]["counter"] = {"niter-out": 2}
    dt = config["simulation"]["dt"]
    byhand, byloop = (pg.get_solver(config, mesh, backend(config)) for _ in range(2))
    for mb in (byhand, byloop):
        perturb(mb, np.random.default_rng(17))
    for _ in range(3):
        byhand.step(dt)
    byloop.run()
    assert byloop.nrt == byhand.nrt == 3 and byloop.tme.get() == byhand.tme.get()
    for a, b in zip(byhand.blocks, byloop.blocks):
        assert np.array_equal(a.Q.get(), b.Q.get())
        assert np.abs(a.Q.get() - a.Q1.get()).max() > 0.0
    # the plugin acted once, after the second step, and was finalized once
    counter = byloop.plugins["counter"]
    assert counter.acted == [(2, 2 * dt)] and counter.finalized == 1
    assert byhand.plugins["counter"].acted == []


def test_step_marches_a_one_cell_thick_block(backend):
    # the length across the one-cell axis is infinite, so the CFL step is not limited by it; slip walls close that axis
    config, mesh = uniform(
        "navier-stokes", "tvd-rk3", "cfl", dims=(6, 6, 2), periodic=(True, True, False), names={5: "slip", 6: "slip"}
    )
    config["bcvals"]["slip"] = {"kind": "slp-adia-wall"}
    mb = pg.get_solver(config, mesh, backend(config))
    blk = mb.blocks[0]
    lengths = blk.dijk.get()[blk.interior]
    assert np.isinf(lengths[..., 2]).all() and np.isfinite(lengths[..., :2]).all()
    # the fastest mode on the fifth-metre cells of the two marched axes, sqrt(2) not sqrt(3), and no convection across the third
    R = Ru / air["Air"]["MW"]
    c = np.sqrt(air["Air"]["cp0"] / (air["Air"]["cp0"] - R) * R * 300.0)
    dx = 0.2
    speeds = mb.cflmax()
    assert speeds["acoustic"] == pytest.approx(c * np.sqrt(2.0) / dx, rel=1e-12)
    assert speeds["convective"] == pytest.approx(10.0 / dx, rel=1e-12)
    assert speeds["combined"] == pytest.approx(speeds["acoustic"] + speeds["convective"], rel=1e-15)
    dt = mb.integ.controller.stepsize(mb)
    assert dt == min(config["simulation"]["cfl"] / speeds["combined"], config["simulation"]["dt-max"])
    Q0 = blk.Q.get()
    for _ in range(3):
        mb.step(dt)
    assert_unchanged(mb, Q0)


@pytest.mark.parametrize("integrator", schemes)
@pytest.mark.parametrize("physics", ["euler", "navier-stokes"])
@pytest.mark.parametrize("recon", ["constant", "muscl"])
def test_a_periodic_box_keeps_every_conserved_total(backend, integrator, physics, recon):
    # four blocks, so the exchange between blocks telescopes as a block's own periodic faces do
    config, mesh = box(physics=physics, integrator=integrator, dt=2e-5, mbdims=(2, 2, 1), dims=(7, 6, 5))
    if recon == "muscl":
        config["adv-flux"]["recon"], config["adv-flux"]["limiter"] = "muscl", "koren"
    mb = pg.get_solver(config, mesh, backend(config))
    assert mb.ng == (2 if recon == "muscl" else 1)
    perturb(mb, np.random.default_rng(11))
    before, _ = totals(mb)
    Q0 = [blk.Q.get()[blk.interior].copy() for blk in mb.blocks]
    for _ in range(3):
        mb.step(config["simulation"]["dt"])
    after, scale = totals(mb)
    # measured at most 7.6e-16 of the total magnitude on every row; a face counted once or weighted wrong reads 1e-6 and up
    assert (np.abs(after - before) / scale < 1e-13).all()
    # and the state moved: the density by a percent, the momentum by more than the density's size
    moved = np.max([np.abs(blk.Q.get()[blk.interior] - q0).max(axis=(0, 1, 2)) / np.abs(q0[..., 0]).max() for blk, q0 in zip(mb.blocks, Q0)], axis=0)
    assert moved[0] > 1e-3 and (moved[1:4] > 0.1).all()


def test_a_mixture_keeps_every_species_total(backend):
    # the species rows and the diffusion's correction velocity, under the thermally perfect gas
    config, mesh = box(physics="navier-stokes", gas="CH4_O2", dt=2e-5, mbdims=(2, 2, 1), dims=(7, 6, 5))
    mb = pg.get_solver(config, mesh, backend(config))
    perturb(mb, np.random.default_rng(11))
    before, _ = totals(mb)
    for _ in range(3):
        mb.step(config["simulation"]["dt"])
    after, scale = totals(mb)
    # the eos takes a fixed count of Newton steps from a cell's last temperature, so a halo's state sits its floor off its image's and the two blocks' fluxes through one face differ by it: measured 1.0e-13 on the momentum rows, 3e-16 on the rest
    assert (np.abs(after - before) / scale < 1e-12).all()
    assert (scale[5:] > 0).all()


@pytest.mark.parametrize("eos", ["tpg", "real-gas"])
def test_a_dense_gas_box_keeps_every_conserved_total(backend, eos):
    # a perturbed dense cold state, far from ideal for the cubic
    config, mesh = box(integrator="tvd-rk3", dt=1e-7)
    config["mixture"]["species"] = ["O2", "N2", "CO2", "CH4"]
    config["mixture"]["eos"] = eos
    config["mixture"]["trans"] = None
    config["mixture"]["T-range"] = (300.0, 3500.0)
    config["ics"].update(p=60e5, T=320.0, Y={"O2": 0.3, "N2": 0.4, "CO2": 0.2})
    mb = pg.get_solver(config, mesh, backend(config))
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    q[..., 0] *= 1 + 0.01 * np.random.default_rng(3).random(q.shape[:3])
    mb.set_prims([q])
    before, _ = totals(mb)
    Q0 = blk.Q.get()[blk.interior].copy()
    for _ in range(5):
        mb.step(config["simulation"]["dt"])
    Q = blk.Q.get()[blk.interior]
    assert np.isfinite(Q).all()
    after, scale = totals(mb)
    # the eos takes a fixed count of Newton steps from a cell's last temperature, and a halo's start is its own primitives', not its image's, so the two sides of a periodic face sit the eos's floor apart: measured 5e-12 on the momentum rows, 2e-15 on the rest
    assert (np.abs(after - before) / scale < 1e-11).all()
    # the pressure perturbation set the gas moving: the momentum rows by a thousandth of the density's size (measured 1e-3)
    moved = np.abs(Q - Q0).max(axis=(0, 1, 2)) / np.abs(Q0[..., 0]).max()
    assert (moved[1:4] > 1e-5).all()


def face_difference(blk):
    ng = blk.ng
    cells = [n - 1 for n in blk.extents]
    lows, highs = [], []
    for axis, name in enumerate("ijk"):
        F = getattr(blk, f"{name}F").get()
        low = [slice(ng, ng + n) for n in cells]
        high = list(low)
        high[axis] = slice(ng + 1, ng + cells[axis] + 1)
        lows.append(F[tuple(low)])
        highs.append(F[tuple(high)])
    # summed as the apply kernel sums, so the two agree to the bit
    total = lows[0] + lows[1] + lows[2] - highs[0] - highs[1] - highs[2]
    return total * blk.detJinv.get()[blk.interior][..., None]


@pytest.mark.parametrize("physics", ["euler", "navier-stokes"])
@pytest.mark.parametrize("recon", ["constant", "muscl"])
def test_the_right_hand_side_is_the_face_flux_difference(backend, physics, recon):
    # periodic in i and k, walls on j, two blocks each way, so faces of every kind are crossed
    config, mesh = box(physics=physics, dt=1e-6, mbdims=(2, 2, 1), dims=(7, 6, 5), periodic=(True, False, True), names={3: "walls", 4: "walls"})
    config["bcvals"]["walls"] = {"kind": "no-slp-adia-wall" if physics == "navier-stokes" else "slp-adia-wall"}
    if recon == "muscl":
        config["adv-flux"]["recon"], config["adv-flux"]["limiter"] = "muscl", "koren"
    mb = pg.get_solver(config, mesh, backend(config))
    perturb(mb, np.random.default_rng(5))
    stages = mb.integ.stageregs(mb.regs)
    mb.rhs(Q=stages[0].Q, dQ=stages[0].dQ)
    for blk in mb.blocks:
        want = face_difference(blk)
        assert np.abs(want).max() > 0.0
        assert np.array_equal(getattr(blk, stages[0].dQ).get()[blk.interior], want)
    # after a step the stage register holds a state of its own, and its right-hand side is the same identity
    mb.step(config["simulation"]["dt"])
    stage = stages[1]
    mb.rhs(Q=stage.Q, dQ=stage.dQ)
    for blk in mb.blocks:
        assert np.abs(blk.Q1.get() - blk.Q.get()).max() > 0.0
        assert np.array_equal(getattr(blk, stage.dQ).get()[blk.interior], face_difference(blk))
