import numpy as np
import yaml
from copy import deepcopy
from pathlib import Path
from lxml import etree

from peregrine.misc import Progress
from peregrine.writers.base import BaseWriter


class RestartWriter(BaseWriter):
    def __init__(
        self,
        mb,
        iovars,
        fname="q.{n:08d}.h5",
        gridf="g.h5",
        precision="single",
        quiet=True,
        cfg=None,
    ):
        fname = Path(fname)
        super().__init__(mb, str(fname.parent), precision, quiet)
        # the grid the result sits on, relative to the result or absolute
        self.gridf = str(gridf)
        # the case that writes, as its yaml, when there is one
        self.cfgyaml = ""
        if cfg is not None:
            self.cfgyaml = yaml.safe_dump(cfg.to_dict())
        self.iovars = iovars
        # what a result is called, from its step n and time t; set by every write
        self.basename = fname.stem

    @property
    def h5name(self):
        return f"{self.name}.h5"

    @property
    def xmfname(self):
        return f"{self.name}.xmf"

    def _varloc(self, var, nblki):
        return f"{self.h5name}:/results_{nblki:06d}/{var}"

    # a result is written in the grid's blocks, not the partition's pieces
    def _gather_extents(self, mb):
        mine = [(blk.origblk, blk.basebounds) for blk in mb.blocks]
        byrank = self.comm.allgather(mine)
        everyones = [b for blks in byrank for b in blks]

        # a collective round takes one piece per rank, so as many rounds as the most pieces of one block a rank holds
        held = [[b[0] for b in blks] for blks in byrank if blks]
        self.rounds = max((np.bincount(h).max() for h in held), default=1)

        nblocks = max(b[0] for b in everyones) + 1
        extents = np.zeros((nblocks, 3), dtype=np.int32)
        for origblk, (_, i1, _, j1, _, k1) in everyones:
            # the pieces tile the block, so its far corner is the block's; bounds are inclusive node indices
            far = (i1 + 1, j1 + 1, k1 + 1)
            extents[origblk] = np.maximum(extents[origblk], far)

        return extents

    ###########################################################################
    # Writing
    ###########################################################################
    def write(self, mb, nrt, tme):
        self.name = self.basename.format(n=nrt, t=tme)
        names = self.iovars.exported

        qf = self._open(self.h5name)
        self._stamp(qf)
        qf.attrs["nrt"], qf.attrs["tme"] = nrt, tme
        qf.attrs["privars"] = np.array(self.iovars.privars, dtype="S")
        qf.attrs["variables"] = np.array(names, dtype="S")
        qf.attrs["grid"] = self.gridf
        qf.attrs["config"] = self.cfgyaml

        # the file is collective, so every rank creates every dataset; the handles are kept, since a lookup by name is a metadata call on every rank
        dsets = {}
        for nblki, (ni, nj, nk) in enumerate(self.extents):
            results = qf.create_group(f"results_{nblki:06d}")
            cells = (nk - 1, nj - 1, ni - 1)
            for name in names:
                dsets[nblki, name] = results.create_dataset(name, shape=cells, dtype=self.fdtype)

        # one snapshot of each block for the whole write: every export variable
        data = self.iovars.data
        self._host = {id(blk): data(blk, names) for blk in mb.blocks}

        # which of my blocks are pieces of each block of the grid
        mine = {}
        for blk in mb.blocks:
            mine.setdefault(blk.origblk, []).append(blk)

        # the writes are collective, so every rank walks every dataset
        with Progress(len(self.extents), self.quiet) as bar:
            for nblki in range(len(self.extents)):
                for r in range(self.rounds):
                    pieces = mine.get(nblki, ())
                    blk = pieces[r] if r < len(pieces) else None
                    for name in names:
                        self._write_var(dsets[nblki, name], blk, name)
                bar.step(f"Writing out block {nblki}")

        dsets = None
        qf.close()
        self._host = None

        self._refresh_xmf(mb, tme)
        self._write_xmf()

    @staticmethod
    def _offset(blk):
        i0, _, j0, _, k0, _ = blk.basebounds
        return (k0, j0, i0)

    def _write_var(self, dset, blk, name):
        if blk is None:
            self._write_slab(dset)
            return

        ng = blk.ng
        array = self._host[id(blk)][name]
        whole = self.fileorder(array, dset.dtype)
        count = (blk.nk - 1, blk.nj - 1, blk.ni - 1)
        if whole is None:
            # not in file order or the file's type, so gather it first
            view = array[blk.interior].T
            whole = np.ascontiguousarray(view, dtype=dset.dtype)
            srcsel = ((0, 0, 0), count)
        else:
            srcsel = ((ng, ng, ng), count)
        self._write_slab(dset, whole, srcsel, (self._offset(blk), count))

    def _refresh_xmf(self, mb, tme):
        for blkelem in self.gridelem:
            nblki = int(blkelem.get("Name")[1::])
            blkelem.find("Time").set("Value", str(tme))
            for attrelem in blkelem.findall("Attribute"):
                for itemelem in attrelem.iter("DataItem"):
                    # a Function item holds others, it has no location of its own
                    if len(itemelem) > 0:
                        continue
                    var = itemelem.text.split("/")[-1]
                    itemelem.text = self._varloc(var, nblki)

    ###########################################################################
    # The xdmf a result needs beyond a grid's
    ###########################################################################
    def _build_blktpl(self):
        super()._build_blktpl()

        timeelem = etree.SubElement(self.blktpl, "Time")
        timeelem.set("Value", "0.0")

        self.scalartpl = etree.Element("Attribute")
        self.scalartpl.set("Name", "var name here")
        self.scalartpl.set("ScalarAttributeType", "Scalar")
        self.scalartpl.set("Center", "Cell")

        self.vectortpl = etree.Element("Attribute")
        self.vectortpl.set("Name", "vector name here")
        self.vectortpl.set("AttributeType", "Vector")
        self.vectortpl.set("Center", "Cell")
        fnelem = etree.SubElement(self.vectortpl, "DataItem")
        fnelem.set("ItemType", "Function")
        fnelem.set("Function", "JOIN($0, $1, $2)")
        fnelem.set("Dimensions", "Dimension here 3")

        self.itemtpl = etree.Element("DataItem")
        self.itemtpl.set("NumberType", "Float")
        self.itemtpl.set("Dimensions", "var dim nums here")
        nbytes = "8" if self.precision == "double" else "4"
        self.itemtpl.set("Precision", nbytes)
        self.itemtpl.set("Format", "HDF")
        self.itemtpl.text = "result file location here"

    def _decorate(self, blkelem, nblki, ni, nj, nk):
        names = self.iovars.exported
        for var in self.iovars.scalars_among(names):
            self._add_scalar(blkelem, var, nblki, ni, nj, nk)
        for vecname, vnames in self.iovars.vectors_among(names).items():
            self._add_vector(blkelem, vecname, vnames, nblki, ni, nj, nk)

    def _add_scalar(self, blkelem, var, nblki, ni, nj, nk):
        attrelem = deepcopy(self.scalartpl)
        attrelem.set("Name", var)

        itemelem = deepcopy(self.itemtpl)
        itemelem.set("Dimensions", f"{nk - 1} {nj - 1} {ni - 1}")
        itemelem.text = self._varloc(var, nblki)

        attrelem.append(itemelem)
        blkelem.append(attrelem)

    def _add_vector(self, blkelem, vecname, vnames, nblki, ni, nj, nk):
        attrelem = deepcopy(self.vectortpl)
        attrelem.set("Name", vecname)
        fnelem = attrelem.find("DataItem")
        fnelem.set("Dimensions", f"{nk - 1} {nj - 1} {ni - 1} 3")

        for var in vnames:
            itemelem = deepcopy(self.itemtpl)
            itemelem.set("Dimensions", f"{nk - 1} {nj - 1} {ni - 1}")
            itemelem.text = self._varloc(var, nblki)
            fnelem.append(itemelem)

        blkelem.append(attrelem)
