from peregrine.multiblock.collections.interface import BaseInterfaceCollection


class BcInterfaceCollection(BaseInterfaceCollection):
    boundary = True

    def __init__(self, kind, blocks, backend):
        super().__init__(blocks, backend)
        self.name, self.kind = kind.name, kind

    def launches(self, hook, Q):
        launches = []
        for kernel in self.kind.kernels(hook, Q):
            mine = [i for i in self.entries if i.axis == kernel.axis]
            policy = kernel.policy
            tiling = self.table.subtiling((hook, kernel.axis), mine, policy.haloranges, policy.tilekind, kernel.ncomps, name=hook)
            launches.append((kernel, self.table, tiling))
        return launches

    def selects(self, interface):
        return interface.nbr is None and interface.bcname == self.name

    @property
    def columns(self):
        return self.kind.columns

    def start(self, tme):
        for interface in self.entries:
            for spec in self.columns.values():
                interface.alloc(spec)
        self.kind.start(tme, self.entries)
