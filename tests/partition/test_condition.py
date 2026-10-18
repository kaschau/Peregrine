import numpy as np
import pytest

from peregrine.partition import Conditioner
from peregrine.tools.verify import verify

from tests.partition.test_cut import cube, mutual, relabelings


def lattice(**kwargs):
    return cube(mbdims=(2, 2, 1), dims=(5, 5, 5), lengths=(2, 2, 1), **kwargs)


def test_a_lattice_has_a_plane_per_seam():
    mb = lattice()
    planes = Conditioner().planes(mb)
    # the i seam and the j seam, each two pairs of interfaces
    assert sorted(len(pairs) for pairs in planes) == [2, 2]


@pytest.mark.parametrize("perm,flips", list(relabelings()))
def test_a_lattice_merges_to_the_whole_block(perm, flips):
    # block 3 stored any of the 24 ways: merging brings it into its partner's frame
    mb = lattice()
    Conditioner().reorient(mb, mb.get_blk(3), perm, flips)
    assert Conditioner().merge(mb) == 2
    assert len(mb.blocks) == 1
    whole = cube(dims=(9, 9, 5), lengths=(2, 2, 1))
    assert np.array_equal(mb.blocks[0].nodes.get(), whole.blocks[0].nodes.get())


@pytest.mark.parametrize("perm,flips", list(relabelings()))
def test_a_reoriented_pair_merges_to_the_whole_block(perm, flips):
    mb = cube(mbdims=(2, 1, 1), dims=(9, 8, 7), lengths=(2, 1, 1))
    Conditioner().reorient(mb, mb.get_blk(1), perm, flips)
    assert Conditioner().merge(mb) == 1
    whole = cube(dims=(17, 8, 7), lengths=(2, 1, 1))
    assert np.array_equal(mb.blocks[0].nodes.get(), whole.blocks[0].nodes.get())


def test_a_periodic_is_a_boundary_to_a_plane():
    # the interior plane goes; the periodic the pair wraps through stays, joining the one block to itself
    mb = cube(mbdims=(2, 1, 1), dims=(5, 5, 5), lengths=(2, 1, 1), periodic=(True, False, False))
    assert Conditioner().merge(mb) == 1
    (blk,) = mb.blocks
    assert blk.extents == (9, 5, 5)
    for nface, xlate in ((1, [-2, 0, 0]), (2, [2, 0, 0])):
        face = blk.get_intface(nface)
        assert face.intkind == "periodic-xlate" and face.nbr == 0
        assert np.allclose(face.xlate, xlate) and np.allclose(face.rot, np.eye(3))
    # a periodic transverse to the plane stops it: nothing merges
    mb = cube(mbdims=(2, 1, 1), dims=(5, 5, 5), lengths=(2, 1, 1), periodic=(True,) * 3)
    assert Conditioner().merge(mb) == 0 and len(mb.blocks) == 2


def test_match_averages_the_copies_of_a_node():
    # a plane shared by two blocks: each copy moves to the mean, half way
    mb = cube(mbdims=(2, 1, 1), dims=(9, 8, 7), lengths=(2, 1, 1))
    b = mb.get_blk(1)
    nodes = b.nodes.get()
    nodes[0, :, :, 0] += 1e-9
    b.nodes.set(nodes)
    assert Conditioner().snap_nodes(mb) == pytest.approx(0.5e-9, rel=1e-6)
    assert np.array_equal(mb.get_blk(0).nodes.get()[-1], mb.get_blk(1).nodes.get()[0])
    # an edge held by four blocks: one copy moved by 4e-9 is 3e-9 off the mean
    mb = lattice()
    b = mb.get_blk(3)
    nodes = b.nodes.get()
    nodes[0, 0, :, 0] += 4e-9
    b.nodes.set(nodes)
    assert Conditioner().snap_nodes(mb) == pytest.approx(3e-9, rel=1e-6)
    assert Conditioner().snap_nodes(mb) == 0.0


def test_orient_puts_the_longest_extent_on_i():
    c = Conditioner()
    mb = cube(mbdims=(2, 1, 1), dims=(5, 9, 7), lengths=(2, 1, 1), defaultbc="wall")
    assert c._longest(mb.get_blk(0)) == ([1, 2, 0], [False, False, False])
    c.orient(mb)
    assert [b.extents for b in mb.blocks] == [(9, 7, 5)] * 2
    assert verify(mb)
    # an odd relabelling reverses k, so the block stays right handed
    mb = cube(mbdims=(2, 1, 1), dims=(9, 5, 7), lengths=(2, 1, 1), defaultbc="wall")
    before = mb.get_blk(0).nodes.get().copy()
    assert c._longest(mb.get_blk(0)) == ([0, 2, 1], [False, False, True])
    c.orient(mb)
    assert [b.extents for b in mb.blocks] == [(9, 7, 5)] * 2
    expected = np.flip(np.moveaxis(before, [0, 2, 1], (0, 1, 2)), axis=2)
    assert np.array_equal(mb.get_blk(0).nodes.get(), expected)
    assert verify(mb)


@pytest.mark.parametrize("perm,flips", list(relabelings()))
def test_orient_restores_any_storage(perm, flips):
    mb = cube(mbdims=(2, 1, 1), dims=(9, 7, 5), lengths=(2, 1, 1), defaultbc="wall")
    c = Conditioner()
    c.reorient(mb, mb.get_blk(1), perm, flips)
    c.reorient(mb, mb.get_blk(0), perm, flips)
    assert verify(mb)
    c.orient(mb)
    assert [b.extents for b in mb.blocks] == [(9, 7, 5)] * 2
    assert mutual(mb) and verify(mb)


def test_condition_leaves_one_oriented_block():
    mb = lattice(defaultbc="wall")
    Conditioner().condition(mb)
    assert [b.extents for b in mb.blocks] == [(9, 9, 5)]
    assert verify(mb)
