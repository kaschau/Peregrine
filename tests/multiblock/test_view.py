import numpy as np

import peregrine as pg


def test_a_view_is_the_live_memory():
    mb = pg.multiblock.Grid()
    pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[4, 3, 2]).fill(mb)
    blk = mb.blocks[0]
    nodes = blk.nodes.view()
    assert blk.nodes.view() is nodes
    before = blk.nodes.get()
    nodes[..., 0] += 1.0
    after = blk.nodes.get()
    assert np.array_equal(after[..., 0], before[..., 0] + 1.0)
    assert np.array_equal(after[..., 1:], before[..., 1:])
    # a copy is its own memory
    assert after is not nodes and not np.shares_memory(after, nodes)
