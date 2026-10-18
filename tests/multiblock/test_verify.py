import numpy as np
import pytest

import peregrine as pg
from peregrine.partition import Conditioner
from peregrine.tools.verify import faceplane, verify

from tests.partition.test_cut import relabelings


def pair():
    mb = pg.multiblock.Grid()
    pg.mesher.CubeMesher(
        mbdims=[2, 1, 1], blkdims=[4, 4, 4], lengths=[1, 1, 1], bcnames={1: "inlet"}, defaultbc="wall"
    ).fill(mb)
    return mb


def test_a_mesher_grid_verifies():
    assert verify(pair())


def test_a_boundary_needs_a_name():
    mb = pair()
    face = mb.blocks[0].get_intface(1)
    assert face.bcname == "inlet" and verify(mb)
    face.bcname = None
    assert not verify(mb)


@pytest.mark.parametrize("perm,flips", list(relabelings()))
def test_a_reoriented_block_still_meets_its_neighbor(perm, flips):
    mb = pair()
    a, b = mb.blocks
    Conditioner().reorient(mb, b, perm, flips)
    assert verify(mb)
    # the tool's hand tables lay the partner's plane out as the interface's own align does
    face = a.get_intface(2)
    ours = np.stack(list(faceplane(a, 2)), axis=-1)
    theirs = np.stack(list(faceplane(b, face.nbrnface)), axis=-1)
    assert np.array_equal(face.align(theirs), ours)
    # within the tolerance of 1e-7 the planes still coincide; past it they do not
    for shift, ok in ((5e-8, True), (1e-6, False)):
        nodes = b.nodes.get()
        nodes[..., 0] += shift
        b.nodes.set(nodes)
        assert verify(mb) == ok


def test_a_left_handed_block_is_refused():
    mb = pg.multiblock.Grid()
    pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[4, 4, 4], defaultbc="wall").fill(mb)
    blk = mb.blocks[0]
    assert verify(mb)
    blk.nodes.set(np.ascontiguousarray(blk.nodes.get()[:, ::-1, :]))
    assert not verify(mb)


@pytest.mark.parametrize("periodic", [False, True])
def test_a_lattice_verifies(periodic):
    mb = pg.multiblock.Grid()
    pg.mesher.CubeMesher(mbdims=[2, 3, 2], blkdims=[3, 3, 3], lengths=[2, 3, 2], periodic=[periodic] * 3, defaultbc="wall").fill(mb)
    assert verify(mb)


def test_a_rotational_periodic_verifies_through_its_rot():
    mb = pg.multiblock.Grid()
    pg.mesher.AnnulusMesher(mbdims=[2, 2, 3], blkdims=[5, 4, 3], sweep=45, periodic=True, defaultbc="wall").fill(mb)
    assert verify(mb)
    # the rot the other way round puts the partner's plane elsewhere
    for blk, face in mb.intfaces():
        if face.rot is not None:
            face.set_periodic(rot=face.rot.T, xlate=face.xlate)
    assert not verify(mb)
