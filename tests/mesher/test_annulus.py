import numpy as np
import pytest

import peregrine as pg
from peregrine.tools.verify import verify

rin, thickness, sweep = 0.5, 0.5, 90.0


def annulus(periodic=True, sweep=sweep, mbdims=[1, 1, 2]):
    mb = pg.multiblock.Grid()
    pg.mesher.AnnulusMesher(
        mbdims=mbdims,
        blkdims=[4, 4, 4],
        p1=[0, 0, 0],
        p2=[1, 0, 0],
        p3=[0, rin, 0],
        sweep=sweep,
        thickness=thickness,
        periodic=periodic,
        defaultbc="wall",
    ).fill(mb)
    return mb


def rodrigues(axis, degrees):
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    th = np.radians(degrees)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * K @ K


def test_nodes_lie_in_the_wedge():
    mb = annulus()
    for blk in mb.blocks:
        nodes = blk.nodes.get()
        # x along the axis, the radius out from it, the angle around it
        assert np.allclose(nodes[..., 0], np.linspace(0, 1, 4)[:, None, None])
        radius = np.linalg.norm(nodes[..., 1:], axis=-1)
        assert np.allclose(radius, np.linspace(rin, rin + thickness, 4)[None, :, None])
        angle = np.degrees(np.arctan2(nodes[..., 2], nodes[..., 1]))
        k = blk.nblki
        assert np.allclose(angle, np.linspace(45 * k, 45 * (k + 1), 4)[None, None, :])
        # right handed: i along the axis, j outward, k around
        o, xi, xj, xk = nodes[0, 0, 0], nodes[1, 0, 0], nodes[0, 1, 0], nodes[0, 0, 1]
        assert np.dot(xk - o, np.cross(xi - o, xj - o)) > 0


def test_the_sweeps_ends_are_turned_onto_each_other():
    mb = annulus()
    a, b = mb.blocks
    # the blocks meet along k, and the far ends meet through the sweep
    assert a.get_intface(6).nbr == 1 and a.get_intface(6).intkind == "interior"
    assert b.get_intface(5).nbr == 0 and b.get_intface(5).intkind == "interior"
    low, high = a.get_intface(5), b.get_intface(6)
    assert low.nbr == 1 and high.nbr == 0
    assert low.intkind == high.intkind == "periodic-rot"
    # the low end's halo comes from the far end turned back by the sweep, the high end's turned forward
    assert np.allclose(low.rot, rodrigues([1, 0, 0], -sweep), atol=1e-15)
    assert np.allclose(high.rot, rodrigues([1, 0, 0], sweep), atol=1e-15)
    assert np.allclose(low.xlate, 0) and np.allclose(high.xlate, 0)
    # the rot carries the high end's node plane onto the low end's
    theirs = b.nodes.get()[:, :, -1].reshape(-1, 3)
    mine = a.nodes.get()[:, :, 0].reshape(-1, 3)
    assert np.allclose(theirs @ low.rot.T, mine, atol=1e-15)
    assert verify(mb)


def test_a_full_turn_closes_on_itself():
    mb = annulus(sweep=360.0)
    a, b = mb.blocks
    for face in (a.get_intface(5), b.get_intface(6)):
        assert face.intkind == "interior" and face.rot is None
    assert a.get_intface(5).nbr == 1 and b.get_intface(6).nbr == 0
    assert np.allclose(a.nodes.get()[:, :, 0], b.nodes.get()[:, :, -1])
    assert verify(mb)


def test_an_unperiodic_sweep_has_named_ends():
    mb = annulus(periodic=False)
    a, b = mb.blocks
    for face in (a.get_intface(5), b.get_intface(6)):
        assert face.nbr is None and face.intkind is None and face.bcname == "wall"
    assert verify(mb)


@pytest.mark.parametrize("bad", [dict(p2=[0, 0, 0]), dict(p3=[0, 0, 0]), dict(p3=[1, 0, 0]), dict(sweep=0), dict(sweep=361), dict(p1=[0, 1, 0], periodic=True)])
def test_a_wedge_that_cannot_be_swept_is_refused(bad):
    with pytest.raises(ValueError):
        pg.mesher.AnnulusMesher(**bad)
