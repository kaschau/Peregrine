import h5py
import numpy as np

from peregrine.misc import Progress
from peregrine.misc import get_comm_rank_size


class GridReader:
    def __init__(self, fname, ranks=None, quiet=True):
        self.fname = fname
        self.quiet = quiet
        with h5py.File(self.fname, "r") as self.f:
            self.nblocks = int(self.f.attrs["nblocks"])
            # (ranks, ranks_per_node) of every partition the grid carries
            names = self.f.get("partitions", ())
            pairs = (tuple(int(x) for x in n.split("x")) for n in names)
            self.partitions = sorted(pairs)
            # which partition we picked, and what the rest of the reads follow
            self.pname, self._cuts, self._rankof = None, None, None
            self.mine = range(self.nblocks)
            if ranks is not None:
                self.mine = self._partition(*ranks)

    def fill(self, mb):
        bar = Progress(len(self.mine), self.quiet)
        with h5py.File(self.fname, "r") as self.f, bar:
            for nblki in self.mine:
                blk = mb.add_blk(nblki)
                coords, extents = self._baseinfo(blk)
                blk.set_extents(*extents)
                # a base multiblock's block holds no coordinates; a grid's takes them
                if "nodes" in getattr(blk, "declared", ()):
                    nodes = blk.nodes.get()
                    slab = blk.nodeslab
                    for c, name in enumerate(("x", "y", "z")):
                        nodes[blk.interior + (c,)] = coords[name][slab].T
                    blk.nodes.set(nodes)
                bar.step(f"Reading in block {nblki}")
            self._read_conn(mb)

    def _partition(self, size, ranks_per_node):
        layouts = [rpn for n, rpn in self.partitions if n == size]
        if not layouts:
            raise ValueError(
                f"this grid carries no {size} rank partition, only "
                f"{['%dx%d' % p for p in self.partitions]}. Balance it with\n"
                f"  peregrine partition {self.fname} -ranks {size}"
                f" -ranks_per_node {ranks_per_node}"
            )
        # another node layout for the same rank count is used rather than refused
        if ranks_per_node not in layouts:
            ranks_per_node = layouts[0]

        name = f"{size}x{ranks_per_node}"
        group = self.f[f"partitions/{name}"]
        rank = np.array(group["rank"])
        cuts = np.array(group["cuts"]) if "cuts" in group else None
        if cuts is None:
            assert len(rank) == self.nblocks, (
                f"the {name} partition covers {len(rank)} blocks, "
                f"but this grid has {self.nblocks}"
            )
        else:
            assert len(rank) == len(cuts), (
                f"the {name} partition has {len(rank)} ranks for "
                f"{len(cuts)} pieces"
            )
        self.pname, self._cuts, self._rankof = name, cuts, rank
        myrank = get_comm_rank_size()[1]
        return [int(n) for n in np.flatnonzero(rank == myrank)]

    def _baseinfo(self, blk):
        if self._cuts is None:
            blk.origblk, blk.baseslice = blk.nblki, None
        else:
            origblk, *bounds = self._cuts[blk.nblki]
            blk.origblk = int(origblk)
            blk.baseslice = tuple(int(b) for b in bounds)

        coords = self.f[f"coordinates_{blk.origblk:06d}"]
        if blk.baseslice is None:
            # stored (nk, nj, ni), so the shape is the extents backwards
            nk, nj, ni = coords["x"].shape
        else:
            i0, i1, j0, j1, k0, k1 = blk.baseslice
            ni, nj, nk = i1 - i0 + 1, j1 - j0 + 1, k1 - k0 + 1
        return coords, (int(ni), int(nj), int(nk))

    def _read_conn(self, mb):
        if self._cuts is not None:
            group = self.f[f"partitions/{self.pname}/connectivity"]
        else:
            group = self.f["connectivity"]
        nbr = np.array(group["nbr"])
        orient = group["orient"].asstr()[:]
        bcname = group["bcname"].asstr()[:]
        rot = np.array(group["rot"])
        xlate = np.array(group["xlate"])

        for blk, face in mb.intfaces():
            mine = blk.nblki, face.nface - 1
            # python types, not the numpy scalars hdf5 hands back
            face.bcname = str(bcname[mine]) or None
            face.orient = str(orient[mine]) or None
            n = int(nbr[mine])
            face.nbr = None if n == -1 else n
            # a rot of all zeros is an interface that is not periodic
            if rot[mine].any():
                face.set_periodic(rot=rot[mine], xlate=xlate[mine])
            # without a partition every block is on this rank, whichever it is
            if n == -1:
                face.nbrrank = None
            elif self._rankof is None:
                face.nbrrank = get_comm_rank_size()[1]
            else:
                face.nbrrank = int(self._rankof[n])

        # a cut partition's blocks are its pieces, not the base grid's blocks
        mb.nblocks = self.nblocks if self._cuts is None else len(self._cuts)
