import numpy as np
import pytest

import peregrine as pg

from tests.gases import configure, primitives


def case(backend, gas, seed=23):
    rng = np.random.default_rng(seed)
    config = pg.Config()
    config["adv-flux"]["recon"], config["adv-flux"]["limiter"] = "muscl", "koren"
    configure(config, gas, "navier-stokes")
    config["bcvals"]["wall"] = {"kind": "no-slp-adia-wall"}
    axis = rng.uniform(0.2, 1.0, 3)
    axis /= np.linalg.norm(axis)
    sweep = float(rng.integers(5, 90))
    p3 = np.cross(axis, rng.uniform(0.2, 1.0, 3))
    mesh = pg.mesher.AnnulusMesher(sweep=sweep, p2=axis, p3=p3, periodic=True, defaultbc="wall")
    mb = pg.get_solver(config, mesh, backend(config))
    blk = mb.blocks[0]
    assert blk.ng == 2
    q = primitives(mb, blk)
    shape = q.shape[:3]
    q[..., 0] = rng.uniform(101325 * 0.9, 101325 * 1.1)
    q[..., 1:4] = rng.uniform(1.0, 1000.0, shape + (3,))
    q[..., 4] = rng.uniform(300 * 0.9, 300 * 1.1)
    if mb.sim.mixture.ns > 1:
        Y = rng.uniform(0.0, 1.0, mb.sim.mixture.ns - 1)
        q[..., 5:] = Y / np.sum(Y)
    mb.set_prims([q])
    return mb, axis, sweep, p3 / np.linalg.norm(p3)


# yields (halo plane, partner interior plane, turn from partner to halo in sweeps): the low face's halo is the far end turned back, the high face's the near end turned on
def pairs(blk, array, layers):
    ng = blk.ng
    inner = np.s_[ng:-ng, ng:-ng]
    planes = np.moveaxis(array, 2, 0)
    low, high = blk.get_intface(5), blk.get_intface(6)
    for layer in range(layers):
        yield low.halo(array)[layer][inner], planes[-(ng + 1 + layer)][inner], -1.0
        yield high.halo(array)[layer][inner], planes[ng + layer][inner], 1.0


def frames(x, axis):
    axial = x @ axis
    radial = x - axial[..., None] * axis
    er = radial / np.linalg.norm(radial, axis=-1, keepdims=True)
    et = np.cross(axis, er)
    ea = np.broadcast_to(axis, er.shape)
    return np.stack([ea, er, et], axis=-2)


# the checks go through cylindrical components, never the code's own rotation matrix
def cylindrical(x, v, axis):
    return np.einsum("...ij,...j->...i", frames(x, axis), v)


def azimuth(x, axis, start):
    radial = x - (x @ axis)[..., None] * axis
    return np.degrees(np.arctan2(radial @ np.cross(axis, start), radial @ start))


def test_the_halo_is_the_partner_swept(backend):
    mb, axis, sweep, start = case(backend, "air")
    blk = mb.blocks[0]
    xcc = blk.xcc.get()
    for xh, xc, turn in pairs(blk, xcc, blk.ng):
        assert np.allclose(xh @ axis, xc @ axis, rtol=0.0, atol=1e-13)
        rh = np.linalg.norm(xh - (xh @ axis)[..., None] * axis, axis=-1)
        rc = np.linalg.norm(xc - (xc @ axis)[..., None] * axis, axis=-1)
        assert np.allclose(rh, rc, rtol=1e-13, atol=0.0)
        swept = (azimuth(xh, axis, start) - azimuth(xc, axis, start) + 180.0) % 360.0 - 180.0
        assert np.allclose(swept, turn * sweep, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("gas", ["air", "CH4_O2"])
def test_the_velocity_and_scalar_gradients_turn_with_the_grid(backend, gas):
    mb, axis, sweep, _ = case(backend, gas)
    blk = mb.blocks[0]
    xcc = blk.xcc.get()
    q = primitives(mb, blk)
    velocity = q[..., 1:4]
    for (xh, xc, _), (vh, vc, _) in zip(pairs(blk, xcc, blk.ng), pairs(blk, velocity, blk.ng)):
        assert np.allclose(cylindrical(xh, vh, axis), cylindrical(xc, vc, axis), rtol=1e-12, atol=1e-12)
    stage = mb.integ.stageregs(mb.regs)[0]
    mb.rhs(Q=stage.Q, dQ=stage.dQ)
    grads = blk.grads.get()
    # one halo plane of the gradients is exchanged, as far as the diffusive flux reads
    for (xh, xc, _), (gh, gc, _) in zip(pairs(blk, xcc, 1), pairs(blk, grads, 1)):
        for row in range(3, grads.shape[-2]):
            scale = np.abs(gc[..., row, :]).max()
            got, want = cylindrical(xh, gh[..., row, :], axis), cylindrical(xc, gc[..., row, :], axis)
            assert np.allclose(got, want, rtol=0.0, atol=1e-12 * scale)


def test_the_velocity_gradient_turns_as_a_tensor(backend):
    mb, axis, sweep, _ = case(backend, "air")
    blk = mb.blocks[0]
    xcc = blk.xcc.get()
    stage = mb.integ.stageregs(mb.regs)[0]
    mb.rhs(Q=stage.Q, dQ=stage.dQ)
    grads = blk.grads.get()
    for (xh, xc, _), (gh, gc, _) in zip(pairs(blk, xcc, 1), pairs(blk, grads, 1)):
        # the velocity block in each cell's own cylindrical frame, E G E^T
        Eh, Ec = frames(xh, axis), frames(xc, axis)
        got = np.einsum("...ij,...jk,...lk->...il", Eh, gh[..., :3, :], Eh)
        want = np.einsum("...ij,...jk,...lk->...il", Ec, gc[..., :3, :], Ec)
        assert np.allclose(got, want, rtol=0.0, atol=1e-12 * np.abs(want).max())
