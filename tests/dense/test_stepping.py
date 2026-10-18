import numpy as np
import pytest

import peregrine as pg

from tests.gases import primitives
from tests.dense import support

species = ["O2", "N2", "CH4"]
pairs = [("chung", "takahashi"), ("entropy-scaling", "binary"), ("chung", "lewis")]
dt = 1e-7


def box(**names):
    return pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[6, 6, 6], lengths=[1e-3] * 3, **names)


def perturb(mb):
    rng = np.random.default_rng(3)
    prims = []
    for blk in mb.blocks:
        q = primitives(mb, blk)
        s = q.shape[:3]
        q[..., 0] = 60e6 * (1.0 + 0.05 * rng.random(s))
        q[..., 1:4] = 30.0 * rng.random(s + (3,))
        q[..., 4] = 300.0 * (1.0 + 0.1 * rng.random(s))
        q[..., 5:] = 0.3 + 0.1 * rng.random(s + (2,))
        prims.append(q)
    mb.set_prims(prims)


def assert_finite(blk):
    ng = blk.ng
    # the transport kernel makes only the first halo plane
    for array, depth in ((blk.Q, ng), (blk.state, ng), (blk.trstate, 1)):
        a = array.get()
        assert np.isfinite(a[blk.interior]).all()
        # the halo's edges and corners are never written
        for interface in blk.intfaces:
            for plane in interface.halo(a)[:depth]:
                assert np.isfinite(plane[ng:-ng, ng:-ng]).all()


def totals(blk):
    vol = 1.0 / blk.detJinv.get()[blk.interior]
    return (blk.Q.get()[blk.interior] * vol[..., None]).sum(axis=(0, 1, 2))


@pytest.mark.parametrize("trans, diffusion", pairs)
def test_a_periodic_box_steps_and_keeps_its_totals(backend, trans, diffusion):
    # measured 6e-13 on the momenta, 1e-15 on the rest
    mb = support.solver(backend, species, trans, diffusion, mesh=box(periodic=[True] * 3))
    perturb(mb)
    blk = mb.blocks[0]
    before = totals(blk)
    rho0 = blk.Q.get()[blk.interior][..., 0].copy()
    for _ in range(3):
        mb.step(dt)
    assert_finite(blk)
    after = totals(blk)
    kept = [0, 4, 5, 6]
    assert np.abs(after[kept] / before[kept] - 1.0).max() < 1e-12
    assert np.abs(after[1:4] - before[1:4]).max() < 1e-12 * before[0] * 30.0
    assert np.abs(blk.Q.get()[blk.interior][..., 0] / rho0 - 1.0).max() > 1e-6


@pytest.mark.parametrize("trans, diffusion", pairs)
def test_a_box_through_walls_an_inlet_and_an_exit_steps(backend, trans, diffusion):
    bcvals = {
        "inlet": {"kind": "sub-in-fvt", "u": 20.0, "v": 0.0, "w": 0.0, "T": 300.0, "O2": 0.35, "N2": 0.35},
        "exit": {"kind": "sub-out-fp", "p": 60e6},
        "wall": {"kind": "no-slp-isot-wall", "T": 300.0},
    }
    mesh = box(bcnames={1: "inlet", 2: "exit", 3: "wall", 4: "wall", 5: "wall", 6: "wall"})
    mb = support.solver(backend, species, trans, diffusion, mesh=mesh, bcvals=bcvals)
    perturb(mb)
    blk = mb.blocks[0]
    for _ in range(3):
        mb.step(dt)
    assert_finite(blk)
