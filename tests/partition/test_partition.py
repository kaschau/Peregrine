import numpy as np
import pytest

from peregrine.partition import get_partitioner

from tests.partition.test_cut import cube, mutual


# an i seam is 42 cells, a j seam 48
def lattice():
    return cube(mbdims=(4, 2, 1), dims=(9, 8, 7), lengths=(4, 2, 1), defaultbc="wall")


def assignment(groups, nblks):
    assign = np.full(nblks, -1)
    for rank, group in enumerate(groups):
        assign[group] = rank
    return assign


@pytest.mark.parametrize("kind", ["greedy", "metis", "auto"])
def test_the_lattice_is_placed_on_four_ranks_of_two_nodes(kind):
    mb = lattice()
    p = get_partitioner(kind)
    groups = p.partition(mb, 4, 2)
    assert sorted(b for group in groups for b in group) == list(range(8))
    weights, edges = p.blkwts(mb), p.edges(mb)
    assign = assignment(groups, 8)
    loads = np.bincount(assign, weights=weights, minlength=4)
    assert (loads <= loads.mean() * (1.0 + p.tol)).all()
    # pairing j neighbors cuts the six i seams, 252 cells; pairing i neighbors would cut 276
    assert p.traffic(assign, edges) == 252
    metrics = p.metrics(assign, weights, edges, 4, 2)
    assert metrics["total"] == sum(edges.values()) == 444
    # the two i seams between the node halves alone cross the network
    assert metrics["offnode"] * metrics["total"] / 100 == pytest.approx(84)
    assert metrics["maxtrunk"] == 84
    assert p.partition(lattice(), 4, 2) == groups


def test_polish_caps_a_rank_of_unequal_pieces():
    mb = lattice()
    p = get_partitioner("greedy")
    p.cut(mb, [[0, "i", 1]])
    groups = p.partition(mb, 4, 2)
    assert sorted(b for group in groups for b in group) == list(range(len(mb.blocks)))
    loads = np.bincount(assignment(groups, len(mb.blocks)), weights=p.blkwts(mb), minlength=4)
    assert (loads <= loads.mean() * (1.0 + p.tol)).all()


def test_a_block_too_big_for_a_rank_is_cut_down():
    # one 32 x 8 x 8 block over four ranks at granularity 1: four even i slabs, in the order the cuts leave them
    mb = cube(dims=(33, 9, 9), defaultbc="wall")
    groups = get_partitioner("greedy").partition(mb, 4, 2, granularity=1)
    assert sorted(b for group in groups for b in group) == [0, 1, 2, 3]
    assert [b.ncells for b in mb.blocks] == [512] * 4
    cuts = get_partitioner().cuts(mb)
    assert cuts[:, 0].tolist() == [0] * 4
    assert cuts[:, 1:3].tolist() == [[0, 8], [24, 32], [16, 24], [8, 16]]
    assert (cuts[:, 3:] == [0, 8, 0, 8]).all()
    assert mutual(mb)


def test_ranks_must_fill_whole_nodes():
    with pytest.raises(ValueError, match="whole number of nodes"):
        get_partitioner().partition(lattice(), 3, 2)


def test_fewer_blocks_than_seeds_still_place():
    # the seed picks wrap when the blocks are fewer than the starts
    p = get_partitioner("greedy")
    mb = cube(mbdims=(4, 1, 1), dims=(9, 8, 7), lengths=(4, 1, 1))
    assert p.partition(mb, 2, 1) == [[2, 3], [0, 1]]
    mb = cube(mbdims=(3, 1, 1), dims=(9, 8, 7), lengths=(3, 1, 1))
    assert p.partition(mb, 2, 1) == [[1], [0, 2]]


def test_no_seeds_means_one():
    p = get_partitioner("greedy", nstarts=0)
    assert p.partition(lattice(), 2, 1) == get_partitioner("greedy", nstarts=1).partition(lattice(), 2, 1)
