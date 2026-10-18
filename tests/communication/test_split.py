import numpy as np
import pytest

import peregrine as pg

from tests.communication.test_orientation import ReorientedPair, orientations, unorient
from tests.communication.test_xchg import configured, recons

# five steps of a size the state moves at: 2e-6 for the acoustics, 1e-4 for the diffusion to show
cases = {"euler": 2e-6, "navier-stokes": 1e-4}


def bump(blk, ne):
    x, y, z = (blk.xcc.get()[..., c] for c in range(3))
    b = np.exp(-(((x - 1.0) ** 2 + (y - 0.4) ** 2 + (z - 0.6) ** 2) / 0.1))
    q = np.zeros(x.shape + (ne,))
    q[..., 0] = 101325.0 * (1.0 + 0.05 * b)
    q[..., 1] = 30.0 * np.sin(2 * np.pi * y) * b
    q[..., 2] = -20.0 * np.cos(2 * np.pi * z) * b
    q[..., 3] = 10.0 * b
    q[..., 4] = 300.0 * (1.0 + 0.1 * b)
    q[..., 5] = 0.77 - 0.2 * b
    return q


def stepped(backend, mesh, recon, simulator):
    config = configured(recon, simulator)
    mb = pg.get_solver(config, mesh, backend(config))
    assert mb.ng == recons[recon]
    mb.set_prims([bump(blk, mb.sim.ne) for blk in mb.blocks])
    for _ in range(5):
        mb.step(cases[simulator])
    return mb


_wholes = {}


def whole(backend, recon, simulator):
    if (recon, simulator) not in _wholes:
        mesh = pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[9, 6, 5], lengths=[2, 1, 1], defaultbc="wall")
        mb = stepped(backend, mesh, recon, simulator)
        _wholes[recon, simulator] = mb.blocks[0].Q.get()[mb.blocks[0].interior].copy()
    return _wholes[recon, simulator]


@pytest.mark.parametrize("simulator", list(cases))
@pytest.mark.parametrize("recon", list(recons))
@pytest.mark.parametrize("S", orientations())
def test_a_split_case_steps_as_the_whole(backend, S, recon, simulator):
    # the halves' nodes equal the whole's bitwise: 8 cells over 2 along i
    mesh = ReorientedPair(S, mbdims=[2, 1, 1], blkdims=[5, 6, 5], lengths=[2, 1, 1], defaultbc="wall")
    mb = stepped(backend, mesh, recon, simulator)
    a, b = mb.blocks
    halves = [a.Q.get()[a.interior], unorient(b.Q.get()[b.interior], S)]
    got = np.concatenate(halves, axis=0)
    expected = whole(backend, recon, simulator)
    assert got.shape == expected.shape
    if S == "123":
        # the same cells in the same order: the same bits
        assert np.array_equal(got, expected)
    # a reoriented block sums its fluxes in another axis order: roundoff, 8.8e-16 per row advected and 3.8e-15 diffused
    scale = np.abs(expected).max(axis=(0, 1, 2))
    tolerance = {"euler": 1e-14, "navier-stokes": 2e-14}[simulator]
    assert (np.abs(got - expected).max(axis=(0, 1, 2)) <= tolerance * scale).all()
