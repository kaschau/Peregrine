from types import SimpleNamespace

import numpy as np
import pytest

import peregrine as pg

from tests.gases import same
from tests.simulator import oracle as orc
from tests.navstokes import support

VALUES = {"nu": 0.1, "floor": 0.0}


def case(backend, physics="navier-stokes", **adv):
    config = support.config(support.alike)
    config["simulation"]["simulator"] = physics
    config["adv-flux"].update(adv)
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    return pg.get_solver(config, support.box([9, 8, 7], [1.0, 1.0, 1.0], defaultbc="walls"), backend(config))


def blended(backend, values=VALUES, **adv):
    return case(backend, formulation="blended", secondary="rusanov", switch="ducros", values=values, **adv)


def fluxed(mb):
    mb.launch(mb.kernels["dqdxyz"]("Q"), "interior")
    mb.launch(mb.kernels["advflux"]("Q"), "interior")


def fluxed_linear(mb, A, u0=(10.0, 0.0, 0.0)):
    blk = mb.blocks[0]
    support.setprims(mb, blk, 101325.0, np.array(u0) + blk.xcc.get() @ A.T, 300.0, 0.6)
    fluxed(mb)


def fluxed_smooth(mb, seed=3):
    blk = mb.blocks[0]
    rng = np.random.default_rng(seed)
    X = blk.xcc.get()
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    ph = rng.uniform(0.0, 2.0 * np.pi, 8)
    p = 101325.0 * (1 + 0.2 * np.sin(2 * np.pi * x + ph[0]) * np.cos(2 * np.pi * y + ph[1]))
    u = 60.0 * np.cos(2 * np.pi * x + ph[2]) + 20.0 * np.sin(2 * np.pi * z + ph[3])
    v = 15.0 * np.sin(2 * np.pi * y + ph[4]) + 40.0 * np.cos(2 * np.pi * x + ph[1]) - 25.0
    w = -10.0 * np.cos(2 * np.pi * z + ph[5]) + 30.0 * np.sin(2 * np.pi * y + ph[2])
    T = 300.0 * (1 + 0.3 * np.cos(2 * np.pi * x + ph[6]) * np.sin(2 * np.pi * z + ph[7]))
    Y = (0.5 + 0.3 * np.sin(2 * np.pi * (x + y) + ph[0]))[..., None]
    support.setprims(mb, blk, p, np.stack([u, v, w], axis=-1), T, Y)
    fluxed(mb)


def faces(mb, axis, deep=False):
    blk = mb.blocks[0]
    ng = blk.ng
    a = "ijk"[axis]
    cells = dict(Q=blk.Q.get(), state=blk.state.get(), grads=blk.grads.get(), dijk=blk.dijk.get())
    arrays = dict(cells, S=getattr(blk, f"{a}S").get(), F=getattr(blk, f"{a}F").get())
    # deep keeps one cell off the halo, where a wall's mirror does not reach
    across = slice(ng + 1, -ng - 1) if deep else slice(ng, -ng)
    arrays = {k: np.moveaxis(v, axis, 0)[:, across, across] for k, v in arrays.items()}
    n = arrays["Q"].shape[0] - 2 * ng
    lo, hi = (ng + 2, ng + n - 1) if deep else (ng + 1, ng + n)
    at = {k: {o: arrays[k][lo + o : hi + o] for o in (-1, 0)} for k in cells}
    return SimpleNamespace(S=arrays["S"][lo:hi], F=arrays["F"][lo:hi], **at)


def blend(mb, f, values):
    L, R = orc.cell(f.Q[-1], f.state[-1]), orc.cell(f.Q[0], f.state[0])
    w, contact = orc.ducros(f.grads, f.Q, f.dijk, **values)
    assert np.array_equal(w, contact)
    return w, (1 - w)[..., None] * orc.kepaec(L, R, f.S) + w[..., None] * orc.rusanov(L, R, f.S)


def close(got, want, within):
    scale = np.abs(want).reshape(-1, want.shape[-1]).max(axis=0)
    scale = np.where(scale > 0.0, scale, scale.max())
    assert (np.abs(got - want) <= within * scale).all()


def test_the_blend_is_weighed_by_the_sensor(backend):
    mb = blended(backend)
    fluxed_smooth(mb)
    for axis in range(3):
        f = faces(mb, axis)
        w, want = blend(mb, f, VALUES)
        # the sensor spans its range on this state: measured 0.0 to 0.99, the least along i 0.037
        assert w.min() < 0.05 and w.max() > 0.9
        # measured at most 3.4e-16 of a row's largest
        close(f.F[..., : mb.sim.ne], want, 1e-13)


def test_a_shear_reads_none(backend):
    # du/dy alone: no dilatation, so the sensor is the floor, none, and the blend is kepaec
    A = np.array([[0.0, 30.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mb, alone = blended(backend), case(backend, primary="kepaec")
    fluxed_linear(mb, A)
    fluxed_linear(alone, A)
    for axis in range(3):
        f, fk = faces(mb, axis, deep=True), faces(alone, axis, deep=True)
        w, _ = blend(mb, f, VALUES)
        # the velocity off the solution is rho u over rho, so the dilatation is roundoff and the sensor its square: measured 3e-63
        assert w.max() < 1e-20
        # that times rusanov's: measured 8e-64 of a row's largest
        close(f.F[..., : mb.sim.ne], fk.F[..., : mb.sim.ne], 1e-13)


def test_a_dilatation_reads_one_before_the_high_pass(backend):
    # du/dx = dv/dy = dw/dz alone: no vorticity, the 4/3 clipped to one; at nu 0 the high-pass passes all, so the blend is rusanov
    A = 30.0 * np.eye(3)
    values = {"nu": 0.0, "floor": 0.0}
    mb, alone = blended(backend, values), case(backend, primary="rusanov")
    fluxed_linear(mb, A)
    fluxed_linear(alone, A)
    for axis in range(3):
        f, fr = faces(mb, axis, deep=True), faces(alone, axis, deep=True)
        w, _ = blend(mb, f, values)
        assert (w == 1.0).all()
        assert same(mb, f.F, fr.F)


def test_the_floor_is_the_least_weight(backend):
    # a shear at a floor: the sensor is the floor everywhere
    A = np.array([[0.0, 30.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    values = {"nu": 0.1, "floor": 0.3}
    mb = blended(backend, values)
    fluxed_linear(mb, A)
    for axis in range(3):
        f = faces(mb, axis, deep=True)
        w, want = blend(mb, f, values)
        assert (w == 0.3).all()
        # measured at most 4.7e-15 of a row's largest
        close(f.F[..., : mb.sim.ne], want, 1e-13)


@pytest.mark.parametrize(
    "A,weight",
    [
        # a dilatation of 3 against a vorticity of 1: 4/3 of 9/10, clipped to one
        (np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), 1.0),
        # a dilatation of 1 against a vorticity of 2: 4/3 of 1/5
        (np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), 4.0 / 15.0),
    ],
    ids=["clipped", "sharpened"],
)
def test_the_ratio_is_sharpened_by_the_four_thirds_and_clipped_at_one(backend, A, weight):
    values = {"nu": 0.0, "floor": 0.0}
    mb = blended(backend, values)
    fluxed_linear(mb, 30.0 * A)
    for axis in range(3):
        f = faces(mb, axis, deep=True)
        w, want = blend(mb, f, values)
        # the gradients of a linear field are exact on the box: measured 7.8e-16
        assert np.abs(w - weight).max() < 1e-12
        # measured at most 4.1e-15 of a row's largest
        close(f.F[..., : mb.sim.ne], want, 1e-13)


def test_the_high_pass_kills_a_slow_dilatation(backend):
    # the same dilatation under a flow at rest and under one a hundred times faster than the velocity varies across the box: at the cell's half-length over nu the fast flow's cutoff is forty times the dilatation
    A = 3.0 * np.eye(3)
    weights = []
    for u0 in ((0.0, 0.0, 0.0), (300.0, 0.0, 0.0)):
        mb = blended(backend)
        fluxed_linear(mb, A, u0)
        least, most = 1.0, 0.0
        for axis in range(3):
            f = faces(mb, axis, deep=True)
            w, want = blend(mb, f, VALUES)
            least, most = min(least, w.min()), max(most, w.max())
            # measured at most 5.8e-15 of a row's largest on the host, 1.8e-13 on an H100 under fused multiply-add
            close(f.F[..., : mb.sim.ne], want, 1e-12)
        weights.append((least, most))
    # measured 0.68 to 0.96 at rest, 3.5e-4 under the flow
    assert weights[0][0] > 0.5 and weights[1][1] < 1e-3


def test_the_values_reach_the_kernel(backend):
    values = {"nu": 0.05, "floor": 0.02}
    mb = blended(backend, values)
    fluxed_smooth(mb)
    kernels = mb.kernels["advflux"]("Q").kernels
    assert all("nu = 0.05" in k.text and "least = 0.02" in k.text for k in kernels)
    # the switch reads the cell either side alone
    assert all(k.stencil == 1 for k in kernels)
    mb = blended(backend, {})
    fluxed_smooth(mb)
    kernels = mb.kernels["advflux"]("Q").kernels
    assert all("nu = 0.1" in k.text and "least = 0.0" in k.text for k in kernels)


def test_euler_has_no_gradients_to_read(backend):
    with pytest.raises(NameError, match="ngrad"):
        mb = blended(backend, physics="euler")
        mb.launch(mb.kernels["advflux"]("Q"), "interior")
