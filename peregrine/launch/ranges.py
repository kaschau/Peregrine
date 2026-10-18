class BaseRange:
    # the first interior plane a halo exchange sends, counted in from the interface
    xchgstart = 0

    def __init__(self, blk):
        self.blk = blk
        self.ni, self.nj, self.nk = blk.extents
        self.ng = blk.ng

    @classmethod
    def of(cls, blk, spec):
        return cls(blk)

    @property
    def exts(self):
        raise NotImplementedError

    @property
    def fullexts(self):
        return tuple(n + 2 * self.ng for n in self.exts)

    def full(self):
        return [((0, 0, 0), self.fullexts)]

    # the allocation's corner cell, a halo cell holding nothing: for a kernel run once per block that reads nothing at its cell
    def origin(self):
        return [((0, 0, 0), (1, 1, 1))]

    def interior(self):
        return [((self.ng,) * 3, self.exts)]

    # over the interface proper only: halo edge and corner cells are never generated, nothing reads or writes one
    def _slab(self, intface, first, count):
        ng, n = self.ng, self.exts
        axis = intface.axis
        start, extent = [ng] * 3, list(n)
        low, high = ng - first - count, ng + n[axis] + first
        start[axis] = low if intface.is_low else high
        extent[axis] = count
        return [(tuple(start), tuple(extent))]

    def halo(self, intface, depth=None):
        return self._slab(intface, 0, self.ng if depth is None else depth)

    def halolayer(self, intface, layer):
        return self._slab(intface, layer, 1)

    def all(self):
        return self.interior() + self.halos()

    # the interior and the first layer of every halo: for a kernel whose values are read one layer into the halo
    def shallow(self):
        return self.interior() + [r for i in self.blk.intfaces for r in self.halo(i, 1)]

    def halos(self):
        return [r for i in self.blk.intfaces for r in self.halo(i)]

    # whether a range lies within the halo behind an interface
    def _within(self, rng, intface):
        ((hstart, hextent),) = self.halo(intface)
        start, extent = rng
        return all(hs <= s and s + e <= hs + he for hs, he, s, e in zip(hstart, hextent, start, extent))

    def less(self, name, at):
        # whole slabs whatever depth at gives
        return [r for r in getattr(self, name)() if not any(self._within(r, i) for i, _ in at)]

    # the slab behind each interface, as deep as the range holds and at most the depth given, ng unless given
    def behind(self, name, at):
        held = getattr(self, name)()
        slabs = []
        for i, depth in at:
            for start, extent in held:
                if self._within((start, extent), i):
                    slabs += self.halo(i, min(extent[i.axis], self.ng if depth is None else depth))
        return slabs

    def haloexts(self, intface, depth=None):
        ((_, extent),) = self.halo(intface, depth)
        axis = intface.axis
        plane = tuple(n for d, n in enumerate(extent) if d != axis)
        return (extent[axis], *plane)

    def halomap(self, intface):
        ng, n, d = self.ng, self.exts, intface.axis
        origin = [ng] * 3
        origin[d] = ng - 1 if intface.is_low else ng + n[d]
        return tuple(origin), self._slabaxes(intface, outward=True)

    def interiormap(self, intface):
        ng, n, d, skip = self.ng, self.exts, intface.axis, self.xchgstart
        origin = [ng] * 3
        origin[d] = ng + skip if intface.is_low else ng + n[d] - 1 - skip
        return tuple(origin), self._slabaxes(intface, outward=False)

    # axes are 1-based so that axis 0 can carry a sign
    def _slabaxes(self, intface, outward):
        d = intface.axis
        sign = -1 if intface.is_low == outward else 1
        return (sign * (d + 1),) + tuple(m + 1 for m in intface.planeaxes)


class CellCenterRange(BaseRange):
    @property
    def exts(self):
        return (self.ni - 1, self.nj - 1, self.nk - 1)


class NodeRange(BaseRange):
    # the interface plane is held by both blocks across it, so the exchange starts one plane in
    xchgstart = 1

    @property
    def exts(self):
        return (self.ni, self.nj, self.nk)


class FaceRange(BaseRange):
    def __init__(self, blk, axis):
        super().__init__(blk)
        self.axis = axis

    @classmethod
    def of(cls, blk, spec):
        return cls(blk, spec.axis)

    @property
    def exts(self):
        extents = [self.ni - 1, self.nj - 1, self.nk - 1]
        extents[self.axis] += 1
        return tuple(extents)

    def _cuttable(self, name):
        if name != "interior":
            raise ValueError(f"{name}: the faces' one range cut at interfaces is interior")

    def less(self, name, at):
        self._cuttable(name)
        start, extent = [self.ng] * 3, list(self.exts)
        for interface, _ in at:
            if interface.axis == self.axis:
                extent[self.axis] -= 1
                start[self.axis] += interface.is_low
        return [(tuple(start), tuple(extent))]

    def behind(self, name, at):
        self._cuttable(name)
        return [p for interface, _ in at if interface.axis == self.axis for p in self.plane(interface)]

    def plane(self, intface):
        axis = intface.axis
        assert axis == self.axis, (intface.nface, self.axis)
        ng, n = self.ng, self.exts
        start, extent = [ng] * 3, list(n)
        start[axis] = ng if intface.is_low else ng + n[axis] - 1
        extent[axis] = 1
        return [(tuple(start), tuple(extent))]


class SlotRange:
    def __init__(self, blk):
        self.cells = CellCenterRange(blk)

    def full(self):
        return self.cells.full()


class InterfacePlaneRange(BaseRange):
    def __init__(self, intface):
        super().__init__(intface.blk)
        self.intface = intface
        self.ng = 0

    @classmethod
    def of(cls, intface, spec):
        return cls(intface)

    @property
    def exts(self):
        exts = [n - 1 for n in self.intface.blk.extents]
        exts[self.intface.axis] = 1
        return tuple(exts)
