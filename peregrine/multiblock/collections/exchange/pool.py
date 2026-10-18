import numpy as np

from peregrine.backend.allocation import Subview


class BasePool:
    # the column kind the buffers answer under
    name = None

    def __init__(self, spec, collections, backend):
        self.spec = spec
        carvings, offset = [], 0
        # each rank's part: where it lies in the pool
        self.slices = {}
        for collection in sorted(collections, key=lambda c: c.rank):
            start = offset
            # in an order both ranks compute alike, so the sender's layout is the receiver's
            for interface in sorted(collection.entries, key=self.order):
                shape = self.planes(collection, interface) + spec.comps
                carvings.append((collection, interface, offset, shape))
                offset += int(np.prod(shape))
            self.slices[collection.rank] = slice(start, offset)
        name = f"{self.name}_{spec.name}Pool"
        self.array = backend.alloc((max(offset, 1),), spec.dtype, name=name)
        for collection, interface, start, shape in carvings:
            buffer = Subview(self.array, start, shape, name=self.name)
            bufs = collection.bufs.setdefault((self.name, spec.name), {})
            bufs[interface] = self.inframe(collection, interface, buffer)
        # each rank's part, which a copy of its message alone moves
        self.parts = {}
        for rank, sl in self.slices.items():
            shape = (sl.stop - sl.start,)
            name = f"{self.name} {spec.name} rank {rank}"
            self.parts[rank] = Subview(self.array, sl.start, shape, name=name)

    def order(self, interface):
        raise NotImplementedError

    def planes(self, collection, interface):
        raise NotImplementedError

    def inframe(self, collection, interface, buffer):
        raise NotImplementedError


class SendPool(BasePool):
    name = "sbuf"

    def order(self, interface):
        return (interface.blk.nblki, interface.nface)

    def planes(self, collection, interface):
        return collection.planes(interface, self.spec)

    def inframe(self, collection, interface, buffer):
        return buffer


class RecvPool(BasePool):
    name = "rbuf"

    def order(self, interface):
        return (interface.nbr, interface.nbrnface)

    def planes(self, collection, interface):
        return collection.acrossplanes(interface, self.spec)

    def inframe(self, collection, interface, buffer):
        planes = collection.planes(interface, self.spec)
        ours = collection.remap_across(interface, (0, 0, 0), (1, 2, 3), planes)
        return Subview.along(buffer, *ours, planes, name=self.name)
