from functools import cached_property, partial

from peregrine.backend.allocation import Subview
from peregrine.launch.arrays import ArraySpec, Broadcast
from peregrine.launch.kernel import Kernel
from peregrine.multiblock.collections.interface import BaseInterfaceCollection


class BaseExchangeInterfaceCollection(BaseInterfaceCollection):
    # the template every kernel of the exchange is rendered from
    template = "peregrine.multiblock.collections.exchange.kernels.place"

    def __init__(self, blocks, backend, provider, exchanged):
        super().__init__(blocks, backend)
        # the case's provider, which the placing kernels are made against
        self.provider = provider
        # the exchanged array families, in declaration order, and their banks
        self.exchanged = exchanged = list(exchanged)
        self.arrays = {bank: spec for spec in exchanged for bank in spec.banks}
        # per interface: the interface across on this rank, the rot on the device, and the pools' buffers by column
        self.nbrs, self.rots, self.bufs = {}, {}, {}
        # the columns this collection answers itself, each a view made per interface when asked
        self.views = {"rot": self.rot}
        for bank in self.arrays:
            self.views[bank] = partial(self.haloview, bank=bank)
            self.views["interior", bank] = partial(self.interiorview, bank=bank)
            self.views["nbr", bank] = partial(self.nbrview, bank=bank)

    def _entries(self):
        blocks = {blk.nblki: blk for blk in self.blocks}
        interfaces = super()._entries()
        for interface in interfaces:
            across = blocks.get(interface.nbr)
            nbr = across.get_intface(interface.nbrnface) if across else None
            self.nbrs[interface] = nbr
        return interfaces

    def column(self, name):
        if name in self.views:
            return [self.views[name](i) for i in self.entries]
        if name in self.bufs:
            return [self.bufs[name].get(i) for i in self.entries]
        return super().column(name)

    ###########################################################################
    # The planes an interface trades, and the views onto them
    ###########################################################################
    def depth(self, interface, spec):
        return interface.blk.ng if spec.exchanged is True else spec.exchanged

    def planes(self, interface, spec):
        depth = self.depth(interface, spec)
        return spec.blkrange(interface.blk).haloexts(interface, depth)

    def acrossplanes(self, interface, spec):
        planes = self.planes(interface, spec)
        ours = dict(zip(interface.planeaxes, planes[1:]))
        return planes[:1] + tuple(ours[axis] for axis, _ in interface.nbrplane)

    def rotated(self, interface, spec):
        return interface.rotperiodic and spec.vecstart is not None

    def remap_across(self, interface, origin, axes, planes):
        origin, axes = list(origin), list(axes)
        plane = interface.nbrplane
        out = [axes[0]]
        for ouraxis, extent in zip(interface.planeaxes, planes[1:]):
            m = next(i for i, (axis, _) in enumerate(plane) if axis == ouraxis)
            # a slab map's axes are 1-based and signed, negative running backwards (Subview.along)
            theirs = abs(axes[1 + m]) - 1
            if plane[m][1]:
                origin[theirs] += extent - 1
                out.append(-axes[1 + m])
            else:
                out.append(axes[1 + m])
        return tuple(origin), tuple(out)

    def window(self, interface, blk, bank, slabmap):
        return Subview.along(getattr(blk, bank), *slabmap, self.planes(interface, self.arrays[bank]))

    def haloview(self, interface, bank):
        blk = interface.blk
        return self.window(interface, blk, bank, self.arrays[bank].blkrange(blk).halomap(interface))

    def interiorview(self, interface, bank):
        blk = interface.blk
        return self.window(interface, blk, bank, self.arrays[bank].blkrange(blk).interiormap(interface))

    def nbrview(self, interface, bank):
        nbr = self.nbrs[interface]
        if nbr is None:
            return None
        spec = self.arrays[bank]
        theirs = spec.blkrange(nbr.blk).interiormap(nbr)
        ours = self.remap_across(interface, *theirs, self.planes(interface, spec))
        return self.window(interface, nbr.blk, bank, ours)

    def rot(self, interface):
        if not interface.rotperiodic:
            return None
        if interface not in self.rots:
            self.rots[interface] = self.backend.alloc((3, 3), name="rot")
            self.rots[interface].set(interface.rot)
        return self.rots[interface]

    ###########################################################################
    # The kernels and the nodes
    ###########################################################################
    @property
    def columns(self):
        rot = ArraySpec("rot", Broadcast, (3, 3))
        return {"rot": rot, **self._columns()}

    def _columns(self):
        raise NotImplementedError

    @cached_property
    def placed(self):
        return self._placed()

    def _placed(self):
        raise NotImplementedError

    def _rotating(self, spec):
        rotating = any(self.rotated(i, spec) for i in self.entries)
        return (False, True) if rotating else (False,)

    def _place(self, spec, dst, src, rotate):
        options = dict(comps=spec.comps, rotate=rotate, vecstart=spec.vecstart)
        columns = self.columns
        return Kernel(self.provider, self.template, options=options, columns=columns, dst=dst, src=src, rot="rot")

    def _launches(self, move, bank, interfaces, rotate=False):
        spec = self.arrays[bank]
        key = (spec.name, move, rotate)
        tiling = self.table.subtiling(
            key, interfaces, lambda i: [((0, 0, 0), self.planes(i, spec))], name=move
        )
        kernel = self.placed[move, bank, rotate]
        return [(kernel, self.table, tiling)]

    def lands(self, move, bank):
        spec = self.arrays[bank]
        sides = {False: [], True: []}
        for i in self.entries:
            sides[self.rotated(i, spec)].append(i)
        return [
            self._launches(move, bank, interfaces, rotate)
            for rotate, interfaces in sides.items()
            if interfaces
        ]

    def forget(self, name):
        for column in (name, ("interior", name), ("nbr", name)):
            self.table.forget(column)
