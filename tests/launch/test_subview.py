import numpy as np
import pytest

import peregrine as pg
from peregrine.backend.allocation import Subview


def array(backend):
    mb = pg.multiblock.Grid(backend=backend(pg.Config()))
    pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[5, 4, 3]).fill(mb)
    nodes = mb.blocks[0].nodes
    data = nodes.get()
    i, j, k = np.indices(data.shape[:3])
    data[...] = (100 * i + 10 * j + k)[..., None]
    nodes.set(data)
    return nodes


def test_a_window_past_the_fastest_axis_is_refused(backend):
    nodes = array(backend)
    nk = nodes.shape[2]
    # along k from its last cell: one cell is fine, two run past, where the flat check sees the next line
    Subview.along(nodes, (0, 0, nk - 1), (1, 2, 3), (1, 1, 1))
    with pytest.raises(ValueError, match="past its"):
        Subview.along(nodes, (0, 0, nk - 1), (1, 2, 3), (1, 1, 2))
    # and backwards from the first
    with pytest.raises(ValueError, match="past its"):
        Subview.along(nodes, (0, 0, 0), (1, 2, -3), (1, 1, 2))


def test_a_window_on_a_reversed_window_reads_the_owner(backend):
    nodes = array(backend)
    ni, nj, nk = nodes.shape[:3]
    # the whole block with i reversed, then its first two planes along the reversed i
    reversed_i = Subview.along(nodes, (ni - 1, 0, 0), (-1, 2, 3), (ni, nj, nk))
    first = Subview.along(reversed_i, (0, 0, 0), (1, 2, 3), (2, nj, nk))
    got = first.get()[..., 0]
    want = nodes.get()[ni - 1 : ni - 3 : -1, :, :, 0]
    assert np.array_equal(got, want)
