from functools import cached_property

from peregrine.launch.table import ArrayTable


class BaseCollection:
    def __init__(self, blocks, backend):
        # the multiblock's own list, not a copy: a collection made before the mesh fills sees the blocks it appends
        self.blocks = blocks
        self.backend = backend

    @cached_property
    def entries(self):
        return self._entries()

    def _entries(self):
        raise NotImplementedError

    @cached_property
    def table(self):
        return ArrayTable(self)

    def column(self, name):
        return [entry.tableval(name) for entry in self.entries]

    def forget(self, name):
        self.table.forget(name)
