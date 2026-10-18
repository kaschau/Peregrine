from peregrine.multiblock.block.grid import GridBlock


class RestartBlock(GridBlock):
    def __init__(self, nblki, mb):
        super().__init__(nblki, mb)
        self.nprims = mb.nprims
