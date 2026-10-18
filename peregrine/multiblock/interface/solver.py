from peregrine.multiblock.interface.grid import GridInterface


class SolverInterface(GridInterface):
    def __init__(self, nface, blk, backend):
        super().__init__(nface, blk)
        self.backend = backend

    @property
    def shape(self):
        extents = self.blk.extents
        return tuple(n - 1 for m, n in enumerate(extents) if m != self.axis)

    def alloc(self, spec):
        array = spec.alloc(self, self.backend)
        setattr(self, spec.name, array)
        return array

    def tableval(self, name):
        own = getattr(self, name, None)
        return own if own is not None else self.blk.tableval(name)
