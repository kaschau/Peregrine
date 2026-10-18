from peregrine.multiblock.collections.exchange import BaseExchangeInterfaceCollection


class InteriorInterfaceCollection(BaseExchangeInterfaceCollection):
    def selects(self, interface):
        return interface.nbr is not None and not interface.offrank

    def _columns(self):
        return {("nbr", bank): spec for bank, spec in self.arrays.items()}

    def _placed(self):
        return {
            ("fill", bank, rotate): self._place(spec, bank, ("nbr", bank), rotate)
            for spec in self.exchanged
            for bank in spec.banks
            for rotate in self._rotating(spec)
        }
