import socket
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from h5py import h5fd, h5p, h5s
from lxml import etree

from peregrine._version import __version__
from peregrine.misc import get_comm_rank_size


def git_rev():
    here = Path(__file__).parent
    cmd = ["git", "-C", str(here)]
    opts = dict(capture_output=True, text=True, check=True)
    try:

        def git(*a):
            return subprocess.run([*cmd, *a], **opts).stdout.strip()

        head = git("rev-parse", "--short", "HEAD")
        return head + ("+" if git("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        return ""


class BaseWriter:

    def __init__(self, mb, path="./", precision="single", quiet=True):
        self.path = path
        self.precision = precision
        self.quiet = quiet
        self.fdtype = "float64" if precision == "double" else "float32"

        self.comm, self.rank, self.size = get_comm_rank_size()
        self.extents = self._gather_extents(mb)
        # where a file this writer makes came from
        self.provenance = {
            "peregrine": __version__,
            "commit": git_rev(),
            "host": socket.gethostname(),
            "ranks": self.size,
            "command": " ".join(sys.argv),
        }

        self._dxpl = h5p.create(h5p.DATASET_XFER)
        self._dxpl.set_dxpl_mpio(h5fd.MPIO_COLLECTIVE)

        self._tree = None

    ###########################################################################
    # What the grid is
    ###########################################################################
    def _gather_extents(self, mb):
        mine = [(blk.nblki, blk.ni, blk.nj, blk.nk) for blk in mb.blocks]
        everyones = [blk for blks in self.comm.allgather(mine) for blk in blks]

        extents = np.zeros((len(everyones), 3), dtype=np.int32)
        for nblki, ni, nj, nk in everyones:
            extents[nblki] = ni, nj, nk

        return extents

    @property
    def nblocks(self):
        return len(self.extents)

    ###########################################################################
    # The hdf5 side
    ###########################################################################
    def _open(self, fname, mode="w"):
        path = f"{self.path}/{fname}"
        return h5py.File(path, mode, driver="mpio", comm=self.comm)

    def _stamp(self, f):
        for key, value in self.provenance.items():
            f.attrs[key] = value
        now = datetime.now(timezone.utc)
        f.attrs["written"] = now.isoformat(timespec="seconds")

    def _write_slab(self, dset, src=None, srcsel=None, dstsel=None):
        fspace = dset.id.get_space()

        # a rank with no piece joins the collective write with an empty selection, since skipping it hangs the rest
        if src is None:
            fspace.select_none()
            mspace = h5s.create(h5s.NULL)
            nothing = np.empty(0, dtype=dset.dtype)
            dset.id.write(mspace, fspace, nothing, dxpl=self._dxpl)
            return

        mspace = h5s.create_simple(src.shape)
        mspace.select_hyperslab(*srcsel)
        fspace.select_hyperslab(*dstsel)
        dset.id.write(mspace, fspace, src, dxpl=self._dxpl)

    def _write_table(self, group, name, table):
        dset = group.create_dataset(name, shape=table.shape, dtype=table.dtype)
        # collective from rank 0 alone: a file system that needs MPI_File_sync refuses an independent write
        if self.rank == 0:
            whole = np.ascontiguousarray(table)
            slab = ((0,) * table.ndim, table.shape)
            self._write_slab(dset, whole, slab, slab)
        else:
            self._write_slab(dset)

    @staticmethod
    def fileorder(array, dtype):
        transposed = array.T
        # a narrower file type forces the copy too: HDF5 converts a wider slab independently, which a file system that needs MPI_File_sync refuses
        fits = transposed.flags["C_CONTIGUOUS"] and transposed.dtype == dtype
        return transposed if fits else None

    ###########################################################################
    # The xdmf side
    ###########################################################################
    @property
    def tree(self):
        if self._tree is None:
            self._build_tree()
        return self._tree

    @property
    def gridelem(self):
        if self._tree is None:
            self._build_tree()
        return self._gridelem

    def _build_tree(self):
        self._tree = etree.Element("Xdmf")
        self._tree.set("Version", "2")
        domainelem = etree.SubElement(self._tree, "Domain")
        self._gridelem = etree.SubElement(domainelem, "Grid")
        self._gridelem.set("Name", "PEREGRINE Output")
        self._gridelem.set("GridType", "Collection")
        self._gridelem.set("CollectionType", "Spatial")

        self._build_blktpl()
        for nblki, (ni, nj, nk) in enumerate(self.extents):
            self._add_blkelem(nblki, ni, nj, nk)

    def _build_blktpl(self):
        self.blktpl = etree.Element("Grid")
        self.blktpl.set("Name", "B#Here")

        topoelem = etree.SubElement(self.blktpl, "Topology")
        topoelem.set("TopologyType", "3DSMesh")
        topoelem.set("NumberOfElements", "Num Elem Here")
        geomelem = etree.SubElement(self.blktpl, "Geometry")
        geomelem.set("GeometryType", "X_Y_Z")

        dataelem = etree.SubElement(geomelem, "DataItem")
        dataelem.set("NumberType", "Float")
        dataelem.set("Dimensions", "XYZ Dims Here")
        dataelem.set("Precision", "8" if self.precision == "double" else "4")
        dataelem.set("Format", "HDF")
        dataelem.text = "grid file location here"
        for _ in range(2):
            geomelem.append(deepcopy(dataelem))

    def _add_blkelem(self, nblki, ni, nj, nk):
        blkelem = deepcopy(self.blktpl)
        blkelem.set("Name", f"B{nblki:06d}")
        blkelem.find("Topology").set("NumberOfElements", f"{nk} {nj} {ni}")

        for i, coord in enumerate(["x", "y", "z"]):
            coordelem = blkelem.find("Geometry")[i]
            coordelem.set("Dimensions", f"{nk} {nj} {ni}")
            coordelem.text = self._gridloc(coord, nblki)

        self._decorate(blkelem, nblki, ni, nj, nk)
        self._gridelem.append(blkelem)

    def _decorate(self, blkelem, nblki, ni, nj, nk):
        pass

    def _gridloc(self, coord, nblki):
        return f"{self.gridf}:/coordinates_{nblki:06d}/{coord}"

    def _write_xmf(self):
        if self.rank != 0:
            return
        et = etree.ElementTree(self.tree)
        et.write(
            f"{self.path}/{self.xmfname}",
            pretty_print=True,
            encoding="UTF-8",
            xml_declaration=True,
        )
