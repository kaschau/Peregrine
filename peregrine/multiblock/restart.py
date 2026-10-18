from peregrine.launch.ranges import CellCenterRange
from peregrine.multiblock.grid import Grid
from peregrine.multiblock.block import RestartBlock


# holds no step or time: those are the solver's clock, or carried from reader to writer by a tool
class Restart(Grid):
    def _new_blk(self, nblki):
        return RestartBlock(nblki, self)

    def __init__(self, nprims, backend=None):
        super().__init__(backend=backend)
        self.nprims = nprims
        self.decl_array("q", CellCenterRange, comps=self.nprims)
