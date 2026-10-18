import numpy as np
import pytest

import peregrine as pg
from peregrine.partition import get_partitioner

from tests.communication.test_orientation import ReorientedPair
from tests.communication.test_split import stepped
from tests.communication.test_xchg import recons


def files(tmp_path, mesher, ops):
    mb = pg.multiblock.Grid()
    mesher.fill(mb)
    whole = str(tmp_path / "whole.h5")
    pg.writers.GridWriter(mb, whole, precision="double").write(mb)
    cut = str(tmp_path / "cut.h5")
    pg.writers.GridWriter(mb, cut, precision="double").write(mb)
    get_partitioner().cut(mb, ops)
    # the pieces' partition replaces the base one; a writer on the cut grid sizes its tables by the pieces
    pg.writers.GridWriter(mb, cut).add_partition(mb, [[b.nblki for b in mb.blocks]], 1)
    return whole, cut, len(mb.blocks)


cases = {
    "one-i": (None, [[0, "i", 2]], 3),
    "pair-123-j": ("123", [[0, "j", 1]], 4),
    "pair-162-j": ("162", [[0, "j", 1]], 4),
    "pair-231-j": ("231", [[0, "j", 1]], 4),
}


@pytest.mark.parametrize("recon", list(recons))
@pytest.mark.parametrize("case", list(cases))
def test_pieces_step_as_their_blocks(backend, tmp_path, case, recon):
    S, ops, npieces = cases[case]
    if S is None:
        mesher = pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[9, 6, 5], lengths=[2, 1, 1], defaultbc="wall")
    else:
        mesher = ReorientedPair(S, mbdims=[2, 1, 1], blkdims=[5, 6, 5], lengths=[2, 1, 1], defaultbc="wall")
    whole, cut, n = files(tmp_path, mesher, ops)
    assert n == npieces
    base = stepped(backend, pg.readers.GridReader(whole), recon, "euler")
    pieces = stepped(backend, pg.readers.GridReader(cut, ranks=(1, 1)), recon, "euler")
    assert pieces.mesh.pname == "1x1" and len(pieces.blocks) == npieces
    assert sum(b.ncells for b in pieces.blocks) == sum(b.ncells for b in base.blocks)
    for blk in pieces.blocks:
        i0, i1, j0, j1, k0, k1 = blk.basebounds
        ref = base.get_blk(blk.origblk)
        slab = ref.Q.get()[ref.interior][i0:i1, j0:j1, k0:k1]
        assert slab.shape == blk.Q.get()[blk.interior].shape
        assert np.array_equal(blk.Q.get()[blk.interior], slab)
