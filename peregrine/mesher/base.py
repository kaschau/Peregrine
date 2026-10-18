import numpy as np

from peregrine.misc import get_comm_rank_size


class BaseMesher:
    name = None
    # a mesher makes its grid: it came from no file and no partition of one
    fname = None
    pname = None

    def __init__(self, mbdims=[1, 1, 1], blkdims=[10, 10, 10], bcnames={}, defaultbc=None):
        self.mbdims = [int(n) for n in mbdims]
        self.blkdims = list(blkdims)
        # boundaries are named by nface, the rest by the default, for bcvals to say what they are; one left unnamed has no kind and a solver refuses it
        self.bcnames = dict(bcnames)
        self.defaultbc = defaultbc

    @property
    def nblks(self):
        return int(np.prod(self.mbdims))

    def fill(self, mb):
        if mb.blocks:
            raise ValueError(
                f"a mesher fills an empty multiblock, this one already has "
                f"{len(mb.blocks)} blocks"
            )
        for _ in range(self.nblks):
            mb.add_blk()
        mb.nblocks = self.nblks

        mbdims = self.mbdims
        rank = get_comm_rank_size()[1]
        for k in range(mbdims[2]):
            for j in range(mbdims[1]):
                for i in range(mbdims[0]):
                    nblki = k * mbdims[1] * mbdims[0] + j * mbdims[0] + i
                    blk = mb.blocks[nblki]

                    self.shape_blk(blk, i, j, k)
                    self.connect(blk, nblki, (i, j, k))
                    for face in blk.intfaces:
                        # a mesher's blocks are all on this rank, whichever it is
                        face.nbrrank = rank if face.nbr is not None else None
                    self.set_periodics(blk, i, j, k)

    ###########################################################################
    # What a kind of mesh fills in
    ###########################################################################
    @property
    def periodicaxes(self):
        return (False, False, False)

    def shape_blk(self, blk, i, j, k):
        raise NotImplementedError

    def set_periodics(self, blk, i, j, k):
        pass

    ###########################################################################
    # The lattice every kind of mesh is laid out on
    ###########################################################################
    def connect(self, blk, nblki, ijk):
        mbdims, periodic = self.mbdims, self.periodicaxes
        for axis, (n, index, wraps) in enumerate(zip(mbdims, ijk, periodic)):
            stride = int(np.prod(mbdims[:axis]))
            for nface, end, step in ((2 * axis + 1, 0, -1), (2 * axis + 2, n - 1, 1)):
                face = blk.get_intface(nface)
                face.orient = "123"
                if index != end:
                    face.nbr = nblki + step * stride
                elif wraps:
                    # periodic with the identity transform until set_periodics says otherwise
                    face.set_periodic()
                    face.nbr = nblki - step * stride * (n - 1)
                else:
                    face.bcname = self.bcnames.get(nface, self.defaultbc)
                    face.nbr = None
                    face.orient = None
