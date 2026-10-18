from peregrine.multiblock.collections.base import BaseCollection


class BaseInterfaceCollection(BaseCollection):
    # True for a collection of boundaries, whose halos the hooks set
    boundary = False

    def _entries(self):
        return [i for b in self.blocks for i in b.intfaces if self.selects(i)]

    def selects(self, interface):
        raise NotImplementedError

    def depth(self, interface, spec):
        # None: every plane behind the interface
        return None

    def interfaces(self, blk, spec):
        return [(i, self.depth(i, spec)) for i in self.entries if i.blk is blk]
