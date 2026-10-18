from peregrine.multiblock.collections.exchange import BaseExchangeInterfaceCollection


class MpiInterfaceCollection(BaseExchangeInterfaceCollection):
    def __init__(self, rank, blocks, backend, provider, exchanged):
        super().__init__(blocks, backend, provider, exchanged)
        self.rank = rank

    def selects(self, interface):
        return interface.offrank and interface.nbrrank == self.rank

    def _columns(self):
        columns = {}
        for bank, spec in self.arrays.items():
            columns["interior", bank] = spec
        for spec in self.exchanged:
            columns["sbuf", spec.name] = spec
            columns["rbuf", spec.name] = spec
        return columns

    def _placed(self):
        placed = {}
        for spec in self.exchanged:
            sbuf, rbuf = ("sbuf", spec.name), ("rbuf", spec.name)
            for bank in spec.banks:
                pack = self._place(spec, sbuf, ("interior", bank), False)
                placed["pack", bank, False] = pack
                for rotate in self._rotating(spec):
                    unpack = self._place(spec, bank, rbuf, rotate)
                    placed["unpack", bank, rotate] = unpack
        return placed

    def packs(self, bank):
        return [self._launches("pack", bank, self.entries)]
