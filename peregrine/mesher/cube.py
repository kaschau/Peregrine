import numpy as np

from peregrine.mesher.base import BaseMesher


class CubeMesher(BaseMesher):
    name = "cube"

    def __init__(
        self,
        origin=[0, 0, 0],
        lengths=[1, 1, 1],
        periodic=[False, False, False],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.origin = list(origin)
        self.lengths = list(lengths)
        self.periodic = list(periodic)

        # where each block of the lattice starts and ends
        axes = zip(origin, lengths, self.mbdims)
        self.edges = [np.linspace(o, o + L, n + 1) for o, L, n in axes]

    @property
    def periodicaxes(self):
        return tuple(self.periodic)

    def shape_blk(self, blk, i, j, k):
        corner = [e[m] for e, m in zip(self.edges, (i, j, k))]
        lengths = [e[m + 1] - e[m] for e, m in zip(self.edges, (i, j, k))]
        self._cube(blk, corner, lengths, self.blkdims)

    def set_periodics(self, blk, i, j, k):
        for face in blk.intfaces:
            if face.intkind != "periodic-xlate":
                continue
            # the low interface takes its halo from the high end of the axis, so it is moved back down; the high one the other way
            axis = face.axis
            span = -self.lengths[axis] if face.is_low else self.lengths[axis]
            xlate = [span if n == axis else 0.0 for n in range(3)]
            face.set_periodic(xlate=xlate)

    def _cube(self, blk, origin, lengths, dims):
        blk.set_extents(*dims)

        axes = zip(origin, lengths, dims)
        x, y, z = (np.linspace(o, o + L, n) for o, L, n in axes)

        nodes = blk.nodes.get()
        grid = np.meshgrid(x, y, z, indexing="ij")
        nodes[blk.interior] = np.stack(grid, axis=-1)
        blk.nodes.set(nodes)
