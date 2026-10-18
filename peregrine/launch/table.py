import numpy as np

from peregrine.backend import abi


class Tiling:
    def __init__(self, twin, arrays, tilekind, name):
        # arrays holds the uploads the twin points into, alive as long as the tiling
        self.abi, self.arrays, self.tilekind, self.name = twin, arrays, tilekind, name

    @property
    def count(self):
        return self.abi.count

    @property
    def tiles(self):
        return self.abi.tiles

    @property
    def tile(self):
        return self.abi.tile


class ArrayTable:
    def __init__(self, collection):
        self.collection = collection
        # the list itself, not a copy: a block table is over the blocks as they come
        self.entries = collection.entries
        self.backend = collection.backend
        self._columns = {}
        # every upload kept: a device graph built over it holds a column's address, and the same content under a name takes it back
        self._uploaded = {}
        self.tilings = {}

    def forget(self, name):
        self._columns.pop(name, None)

    def _upload(self, host):
        raw = np.frombuffer(host, dtype=np.uint8)
        array = self.backend.alloc(raw.shape, np.uint8)
        array.set(raw)
        return array

    @staticmethod
    def _info(value):
        return abi.noarray if value is None else value.info

    def pointer(self, name):
        if name not in self._columns:
            values = self.collection.column(name)
            if all(v is None for v in values):
                raise KeyError(f"no entry of this table has {name}")
            arrinfos = [self._info(v) for v in values]
            host = (type(arrinfos[0]) * len(arrinfos))(*arrinfos)
            key = (name, bytes(host))
            if key not in self._uploaded:
                self._uploaded[key] = self._upload(host)
            self._columns[name] = self._uploaded[key]
        return self._columns[name].ptr

    def tiling(self, kernel, rangename):
        policy, ncomps = kernel.policy, kernel.ncomps
        # the policy's class: a kernel's own policy carries nothing of its own over a named range
        key = (type(policy), kernel.axis, ncomps, rangename)

        def rangefn(entry):
            return getattr(policy.blkrange(entry), rangename)()

        tilekind = policy.tilekind
        return self.subtiling(key, self.entries, rangefn, tilekind, ncomps, name=rangename)

    def subtiling(self, key, entries, rangefn, tilekind="cells", ncomps=1, name=None):
        if key not in self.tilings:
            index = {id(e): n for n, e in enumerate(self.entries)}
            ranges = [
                (index[id(e)], start, extent, ncomps, *offsets)
                for e in entries
                for start, extent, *offsets in rangefn(e)
            ]
            self._tile(key, ranges, tilekind, name)
        return self.tilings[key]

    def _tile(self, key, ranges, tilekind, name=None):
        tile = self.backend.tile(tilekind)
        count = len(ranges)
        table = np.zeros(count, dtype=np.dtype(abi.Range))
        for index, r in enumerate(ranges):
            entry, start, extent, ncomps, *carried = r
            table["start"][index] = start
            table["extent"][index] = (*extent, ncomps)
            table["n"][index] = int(np.prod(extent)) * ncomps
            table["entry"][index] = entry
            if carried and carried[0]:
                table["offset"][index, : len(carried[0])] = carried[0]
        ntiles = -(-table["n"] // tile)
        first = np.zeros(count + 1, dtype=np.int32)
        first[1:] = np.cumsum(ntiles)
        # which range each tile belongs to, found in one read
        owner = np.repeat(np.arange(count, dtype=np.int32), ntiles)
        counts = table["n"].copy()
        arrays = [self._upload(a) for a in (owner, first, counts, table)]
        twin = abi.Tiling(*(a.ptr for a in arrays), count, int(first[-1]), tile)
        tiling = Tiling(twin, arrays, tilekind, str(key[-1]) if name is None else name)
        self.tilings[key] = tiling
        return tiling
