import peregrine as pg
import pytest
from peregrine.partition import get_partitioner


def pair():
    mb = pg.multiblock.Grid()
    pg.mesher.CubeMesher(mbdims=[2, 1, 1], blkdims=[5, 4, 3], lengths=[2, 1, 1]).fill(mb)
    return mb


# the plane cells of a block's six faces; the traded face is an i face
cells, allplanes, tradedplane = 24, 2 * (6 + 8 + 12), 6


@pytest.mark.parametrize(
    "halocost,tradecost,ng",
    [(1.0, 0.0, 1), (0.0, 1.0, 2), (0.25, 0.06, 2), (0.3, 0.1, 1)],
)
def test_the_weight_is_cells_and_halos(halocost, tradecost, ng):
    mb = pair()
    p = get_partitioner("greedy", halocost=halocost, tradecost=tradecost, ng=ng)
    want = cells + ng * (halocost * allplanes + tradecost * tradedplane)
    assert list(p.blkwts(mb)) == pytest.approx([want, want])
    assert p.edges(mb) == {(0, 1): tradedplane}


def test_a_walls_halo_counts_and_a_trades_counts_more():
    mb = pair()
    p = get_partitioner("greedy")
    (weight, _) = p.blkwts(mb)
    walls = get_partitioner("greedy", tradecost=0.0)
    assert weight > walls.blkwts(mb)[0] > cells
