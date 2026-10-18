import numpy as np

import peregrine as pg
from peregrine.launch.kernel import Kernel
from peregrine.launch.ranges import CellCenterRange

from tests.simulator import support as sp

BINS = 3
kernels = f"{__package__}.kernels"


class WithList(pg.multiblock.Solver):
    def _decl_arrays(self):
        super()._decl_arrays()
        self.decl_array("codes", CellCenterRange, comps=BINS, dtype=np.int32)
        self.decl_array("mark", CellCenterRange, dtype=np.int32)

    def _decl_kerns(self):
        super()._decl_kerns()
        options = dict(bins=BINS)
        self.kernels["reset"] = Kernel(self.provider, f"{kernels}.reset", options=options, codes="codes")
        self.kernels["append"] = Kernel(self.provider, f"{kernels}.append", options=options, codes="codes", mark="mark")
        self.kernels["visit"] = Kernel(self.provider, f"{kernels}.visit", options=options, codes="codes", mark="mark")


def listed(mb):
    blk = mb.blocks[0]
    flat = blk.codes.get().reshape(-1, order=mb.backend.order)
    cells = int(np.prod(blk.codes.shape[:3]))
    total, counts = int(flat[0]), flat[1 : 1 + BINS].astype(int)
    starts = [b * cells + BINS + 1 for b in range(BINS)]
    codes = [flat[s : s + c] for s, c in zip(starts, counts)]
    return total, counts, codes


def test_the_listed_cells_are_found_again(backend):
    mb = sp.case(backend, (7, 6, 5), [1.0, 1.0, 1.0], [True, True, True], cls=WithList)
    blk = mb.blocks[0]
    ng = mb.ng
    n0, n1, n2 = blk.codes.shape[:3]
    mb.launch(mb.kernels["reset"], "origin")
    mb.launch(mb.kernels["append"], "interior")
    mb.launch(mb.kernels["visit"], "full")
    total, counts, codes = listed(mb)
    interior = blk.mark.get()[blk.interior]
    assert total == counts.sum() == interior.size
    seen = []
    for b, segment in enumerate(codes):
        i, rest = segment % n0, segment // n0
        j, k = rest % n1, rest // n1
        assert (segment % BINS == b).all()
        assert (i >= ng).all() and (i < n0 - ng).all()
        assert (j >= ng).all() and (j < n1 - ng).all()
        assert (k >= ng).all() and (k < n2 - ng).all()
        seen += segment.tolist()
    assert len(set(seen)) == total
    # every listed cell marked once, the halo untouched
    assert (interior == 1).all()
    mark = blk.mark.get().copy()
    mark[blk.interior] = 0
    assert not mark.any()
    # the list stands until the next reset: a second visit marks each again
    mb.launch(mb.kernels["visit"], "full")
    assert (blk.mark.get()[blk.interior] == 2).all()
    # the reset empties it: nothing is visited
    mb.launch(mb.kernels["reset"], "origin")
    mb.launch(mb.kernels["visit"], "full")
    assert (blk.mark.get()[blk.interior] == 2).all()
    assert listed(mb)[0] == 0
