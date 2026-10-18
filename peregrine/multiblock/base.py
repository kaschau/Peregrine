from peregrine.backend.host import BaseHostBackend
from peregrine.launch.arrays import ArraySpec
from peregrine.readers import GridReader
from peregrine.multiblock.block import BaseBlock


class BaseMultiBlock:
    def __init__(self, backend=None):
        self.blocks = []
        self.nblocks = None
        # where the blocks' arrays are allocated
        self.backend = backend or BaseHostBackend()
        # every array a block holds, by name, as an ArraySpec
        self.arrays = {}

    def decl_array(self, name, kind, comps=(), **spec):
        self.arrays[name] = ArraySpec(name, kind, comps, **spec)

    @property
    def columns(self):
        specs = self.arrays.values()
        return {bank: spec for spec in specs for bank in spec.banks}

    @classmethod
    def from_grid(cls, fname, *args, quiet=True):
        mb = cls(*args)
        GridReader(fname, quiet=quiet).fill(mb)
        return mb

    def intfaces(self):
        for blk in self.blocks:
            for interface in blk.intfaces:
                yield blk, interface

    def nbrintface(self, interface):
        if interface.nbr is None:
            return None
        blk = self.get_blk(interface.nbr)
        return blk.get_intface(interface.nbrnface) if blk else None

    def get_blk(self, nblki):
        # a block's number is its index only when the whole grid is on one rank
        if nblki < len(self.blocks) and self.blocks[nblki].nblki == nblki:
            return self.blocks[nblki]
        for blk in self.blocks:
            if blk.nblki == nblki:
                return blk

    def _new_blk(self, nblki):
        return BaseBlock(nblki)

    def add_blk(self, nblki=None):
        if nblki is None:
            nblki = max((blk.nblki for blk in self.blocks), default=-1) + 1
        blk = self._new_blk(nblki)
        self.blocks.append(blk)
        return blk

    def __repr__(self):
        string = "Base multiblock object:\n"
        string += f"{len(self.blocks)} block(s)\n"
        return string
