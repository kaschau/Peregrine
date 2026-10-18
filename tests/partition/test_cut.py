import itertools

import numpy as np
import pytest

import peregrine as pg
from peregrine.partition import Conditioner, get_partitioner
from peregrine.tools.verify import verify

partitioner = get_partitioner()


def cube(mbdims=(1, 1, 1), dims=(13, 11, 9), lengths=(1, 1, 1), periodic=(False,) * 3, **kwargs):
    mb = pg.multiblock.Grid()
    pg.mesher.CubeMesher(
        mbdims=list(mbdims),
        blkdims=list(dims),
        lengths=list(lengths),
        periodic=list(periodic),
        **kwargs,
    ).fill(mb)
    return mb


def mutual(mb):
    for blk, face in mb.intfaces():
        if face.nbr is None:
            continue
        partner = mb.get_blk(face.nbr).get_intface(face.nbrnface)
        if partner.nbr != blk.nblki:
            return False
    return True


def relabelings():
    for perm in itertools.permutations(range(3)):
        swaps = sum(1 for a in range(3) for b in range(a + 1, 3) if perm[a] > perm[b])
        for flips in itertools.product((False, True), repeat=3):
            # an even parity of swaps and flips keeps the block right handed
            if (swaps + sum(flips)) % 2 == 0:
                yield list(perm), list(flips)


@pytest.mark.parametrize("ncuts", (1, 2, 3))
@pytest.mark.parametrize("axis", ("i", "j", "k"))
def test_cut_tiles_the_block(axis, ncuts):
    base = cube()
    work = cube()
    partitioner.cut(work, [[0, axis, ncuts]])
    assert len(work.blocks) == ncuts + 1
    assert mutual(work)

    axis = "ijk".index(axis)
    # the cube is axis aligned, so the pieces sort along the cut axis's coordinate
    pieces = sorted(work.blocks, key=lambda blk: blk.nodes.get()[..., axis].min())
    unshared = [slice(None)] * 3
    unshared[axis] = slice(1, None)
    joined = np.concatenate(
        [pieces[0].nodes.get()]
        + [p.nodes.get()[tuple(unshared)] for p in pieces[1:]],
        axis=axis,
    )
    assert np.array_equal(joined, base.blocks[0].nodes.get())


@pytest.mark.parametrize("axis", ("i", "j", "k"))
def test_merge_undoes_cut(axis):
    base = cube()
    work = cube()
    partitioner.cut(work, [[0, axis, 3]])

    assert Conditioner().merge(work) == 3
    assert len(work.blocks) == 1
    assert np.array_equal(work.blocks[0].nodes.get(), base.blocks[0].nodes.get())


def test_evenly_spaced_cuts():
    work = cube()
    partitioner.cut(work, [[0, "j", 2]])
    # nj = 11, so the cuts land at int(11*2/3) = 7 then int(11/3) = 3
    assert sorted(blk.nj for blk in work.blocks) == [4, 4, 5]


@pytest.mark.parametrize("perm,flips", list(relabelings()))
def test_cut_path_follows_orientation(perm, flips):
    mb = cube(mbdims=(2, 1, 1), dims=(9, 8, 7), lengths=(2, 1, 1))
    Conditioner().reorient(mb, mb.get_blk(1), perm, flips)
    face = mb.get_blk(0).get_intface(2)

    for d, axis in enumerate("ijk"):
        path = partitioner.cutpath(mb, 0, axis)
        if axis == "i":
            # block 1 is across an i face, so an i cut never reaches it
            assert path == [[0, "i", False]]
            continue
        # the cut arrives on whichever of the neighbor's axes runs along ours, from its far end when that axis runs backwards
        theirs, flip = face.signedaxis(face.orient[d])
        assert path == [[0, axis, False], [1, "ijk"[theirs], flip]]


@pytest.mark.parametrize("axis", ("j", "k"))
@pytest.mark.parametrize("perm,flips", list(relabelings()))
def test_cut_runs_through_a_reoriented_neighbor(perm, flips, axis):
    # two cuts through a pair whose second block is stored any of the 24 ways: six pieces, and the whole merges back to the one block
    mb = cube(mbdims=(2, 1, 1), dims=(9, 8, 7), lengths=(2, 1, 1), defaultbc="wall")
    Conditioner().reorient(mb, mb.get_blk(1), perm, flips)
    partitioner.cut(mb, [[0, axis, 2]])
    assert len(mb.blocks) == 6
    d = "ijk".index(axis)
    pieces = sorted((b for b in mb.blocks if b.origblk == 0), key=lambda b: b.nodes.get()[..., d].min())
    # nj 8: int(16/3) = 5 then int(8/3) = 2 nodes in; nk 7: int(14/3) = 4 then int(7/3) = 2
    assert [b.extents[d] for b in pieces] == {"j": [3, 4, 3], "k": [3, 3, 3]}[axis]
    assert mutual(mb)
    assert verify(mb)
    assert Conditioner().merge(mb) == 3
    whole = cube(dims=(17, 8, 7), lengths=(2, 1, 1))
    assert len(mb.blocks) == 1
    assert np.allclose(mb.blocks[0].nodes.get(), whole.blocks[0].nodes.get())


def test_a_long_neighbor_is_cut_at_its_own_index():
    # the neighbor's cut index is its own extent's share, not ours
    mb = cube(mbdims=(2, 1, 1), dims=(9, 20, 7), lengths=(2, 1, 1), defaultbc="wall")
    Conditioner().reorient(mb, mb.get_blk(1), [0, 2, 1], [False, False, True])
    partitioner.cut(mb, [[0, "j", 2]])
    assert len(mb.blocks) == 6 and mutual(mb) and verify(mb)


def test_cut_runs_through_periodic():
    mb = cube(mbdims=(2, 1, 1), dims=(9, 8, 7), lengths=(2, 1, 1), periodic=(True,) * 3)
    partitioner.cut(mb, [[0, "j", 1]])

    assert len(mb.blocks) == 4
    assert mutual(mb)
    # both halves of a split periodic need its span and axis to find a partner
    for blk, face in mb.intfaces():
        if not face.intkind.startswith("periodic"):
            continue
        assert face.nbr is not None
        assert face.rot is not None
        assert face.xlate is not None


def test_cut_down_runs_through_a_reoriented_neighbor():
    # a pair one cell thick along its seam can only be cut across it, so every cut runs through the differently stored neighbor
    mb = cube(mbdims=(2, 1, 1), dims=(2, 9, 9), lengths=(2, 1, 1), defaultbc="wall")
    c = Conditioner()
    c.reorient(mb, mb.get_blk(1), [0, 2, 1], [False, False, True])
    c.orient(mb)
    assert [b.extents for b in mb.blocks] == [(9, 9, 2)] * 2
    seam = mb.get_blk(0).get_intface(6)
    assert seam.nbr == 1 and seam.orient != "123"
    partitioner.cut_down(mb, 32)
    assert sorted(b.ncells for b in mb.blocks) == [32] * 4
    assert mutual(mb) and verify(mb)


def test_uncut_grid_is_its_own_base():
    mb = cube(mbdims=(2, 2, 1), dims=(9, 8, 7))
    for blk, (origblk, i0, i1, j0, j1, k0, k1) in zip(
        mb.blocks, partitioner.cuts(mb)
    ):
        assert origblk == blk.nblki
        assert (i0, i1, j0, j1, k0, k1) == (
            0,
            blk.ni - 1,
            0,
            blk.nj - 1,
            0,
            blk.nk - 1,
        )


@pytest.mark.parametrize("axis", ("i", "j", "k"))
@pytest.mark.parametrize("mbdims", ((1, 1, 1), (2, 2, 1)))
def test_every_piece_is_found_in_the_block_it_names(mbdims, axis):
    base = cube(mbdims=mbdims, dims=(9, 8, 7))
    work = cube(mbdims=mbdims, dims=(9, 8, 7))
    partitioner.cut(work, [[0, axis, 3]])

    table = partitioner.cuts(work)
    assert len(table) == len(work.blocks)
    for blk, (origblk, i0, i1, j0, j1, k0, k1) in zip(work.blocks, table):
        assert (blk.ni, blk.nj, blk.nk) == (i1 - i0 + 1, j1 - j0 + 1, k1 - k0 + 1)
        assert np.array_equal(
            blk.nodes.get(),
            base.get_blk(origblk).nodes.get()[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1],
        )
    # the pieces of a base block tile it, so the cells add back up
    assert sum(b.ncells for b in work.blocks) == sum(b.ncells for b in base.blocks)


def test_provenance_survives_repeated_cuts():
    base = cube(dims=(13, 11, 9))
    work = cube(dims=(13, 11, 9))
    # cut a block, then cut its pieces again on another axis
    partitioner.cut(work, [[0, "i", 2]])
    partitioner.cut(work, [[0, "k", 1]])

    for blk, (origblk, i0, i1, j0, j1, k0, k1) in zip(
        work.blocks, partitioner.cuts(work)
    ):
        assert origblk == 0
        assert np.array_equal(
            blk.nodes.get(),
            base.blocks[0].nodes.get()[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1],
        )
