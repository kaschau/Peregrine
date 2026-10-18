from pathlib import Path

import numpy as np

from peregrine.partition import BasePartitioner
from peregrine.misc import Progress
from peregrine.writers.base import BaseWriter


class GridWriter(BaseWriter):
    def __init__(self, mb, fname="g.h5", precision="single", quiet=True):
        fname = Path(fname)
        super().__init__(mb, str(fname.parent), precision, quiet)
        self.h5name = fname.name
        self.xmfname = fname.with_suffix(".xmf").name
        # a grid's xdmf sits beside the grid it points at
        self.gridf = self.h5name

    def write(self, mb):
        gf = self._open(self.h5name)
        self._stamp(gf)
        gf.attrs["nblocks"] = self.nblocks

        # the file is collective, so every rank creates every block's datasets
        for nblki, (ni, nj, nk) in enumerate(self.extents):
            coords = gf.create_group(f"coordinates_{nblki:06d}")
            shape = (nk, nj, ni)
            for name in ("x", "y", "z"):
                coords.create_dataset(name, shape=shape, dtype=self.fdtype)

        # the writes are collective, so every rank walks every dataset
        mine = {blk.nblki: blk for blk in mb.blocks}
        with Progress(len(self.extents), self.quiet) as bar:
            for nblki, (ni, nj, nk) in enumerate(self.extents):
                coords = gf[f"coordinates_{nblki:06d}"]
                blk = mine.get(nblki)
                nodes = blk.nodes.get() if blk is not None else None
                for c, name in enumerate(("x", "y", "z")):
                    if blk is None:
                        self._write_slab(coords[name])
                        continue
                    view = nodes[blk.interior + (c,)].T
                    whole = np.ascontiguousarray(view, dtype=self.fdtype)
                    slab = ((0, 0, 0), (nk, nj, ni))
                    self._write_slab(coords[name], whole, slab, slab)
                bar.step(f"Writing out block {nblki}")

        self._write_conn(gf, mb)
        # the base grid is a partition like any other: one rank owning all of it
        base = gf.create_group("partitions/1x1")
        self._write_table(base, "rank", np.zeros(self.nblocks, dtype=np.int32))
        gf.close()

        self._write_xmf()

    def add_partition(self, mb, blkgroups, ranks_per_node):
        name = f"{len(blkgroups)}x{ranks_per_node}"
        rank = np.full(self.nblocks, -1, dtype=np.int32)
        for r, group in enumerate(blkgroups):
            for nblki in group:
                rank[nblki] = r
        assert not (rank == -1).any(), "every block must be owned by a rank"

        gf = self._open(self.h5name, mode="a")
        if f"partitions/{name}" in gf:
            del gf[f"partitions/{name}"]
        group = gf.create_group(f"partitions/{name}")
        self._write_table(group, "rank", rank)

        if any(blk.baseslice is not None for blk in mb.blocks):
            self._write_table(group, "cuts", BasePartitioner.cuts(mb))
            self._write_conn(group, mb)
        gf.close()

    def _write_conn(self, group, mb):
        shape = (self.nblocks, 6)
        # hdf5 has no None: -1 is no neighbor, "" is unset, an all-zero rot is not periodic
        nbr = np.full(shape, -1, dtype=np.int32)
        orient = np.zeros(shape, dtype=object)
        bcname = np.zeros(shape, dtype=object)
        rot = np.zeros(shape + (3, 3), dtype=np.float64)
        xlate = np.zeros(shape + (3,), dtype=np.float64)
        for blk, face in mb.intfaces():
            # an interface's cell in the tables; nface counts from one
            mine = blk.nblki, face.nface - 1
            if face.nbr is not None:
                nbr[mine] = face.nbr
            orient[mine] = face.orient or ""
            bcname[mine] = face.bcname or ""
            if face.rot is not None:
                rot[mine] = face.rot
                xlate[mine] = face.xlate

        if "connectivity" in group:
            del group["connectivity"]
        conn = group.create_group("connectivity")
        self._write_table(conn, "nbr", nbr)
        self._write_table(conn, "rot", rot)
        self._write_table(conn, "xlate", xlate)
        for name, table in (("orient", orient), ("bcname", bcname)):
            # parallel hdf5 has no variable length strings, so size to the longest
            self._write_table(conn, name, table.astype("S"))
