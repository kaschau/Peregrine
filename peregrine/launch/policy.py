import re
from math import prod

from peregrine.codegen.members import ViewMember
from peregrine.misc import subclass_where
from peregrine.launch.arrays import Broadcast
from peregrine.launch.ranges import InterfacePlaneRange, BaseRange, CellCenterRange, FaceRange, SlotRange


###############################################################################
# Offsets
###############################################################################
class BaseOffset:
    # the words after 'at' this offset is written as
    pattern = None
    # the axis of the face array this offset reaches; None for cells, or for the bound array's own
    axis = None
    # True for an offset the range carries, which the kernel names by its key
    carried = False
    # the kind of array this offset reads, None for the policy's own
    kind = None

    @classmethod
    def parse(cls, match, axis):
        return cls()

    # a carried offset reads inward from the interface and needs no halo layer
    @property
    def reach(self):
        return 0

    @property
    def key(self):
        raise NotImplementedError

    def displace(self, intface, layer):
        raise NotImplementedError

    def binding(self, member):
        raise NotImplementedError


class CellOffset(BaseOffset):
    pattern = r"([+-]?)(\d*)([IJK])"

    def __init__(self, offset):
        self.offset = tuple(offset)

    @staticmethod
    def unit(axis, n=1):
        offset = [0, 0, 0]
        offset[axis] = n
        return tuple(offset)

    @classmethod
    def parse(cls, match, axis):
        n = int(match.group(2) or 1) * (-1 if match.group(1) == "-" else 1)
        return cls(cls.unit("IJK".index(match.group(3)), n))

    @property
    def reach(self):
        return max(abs(d) for d in self.offset)

    def binding(self, member):
        di, dj, dk = self.offset
        return f"{member.cname}.at<{di}, {dj}, {dk}>({member.shape}, at_)"


class ZeroOffset(CellOffset):
    def __init__(self):
        super().__init__((0, 0, 0))

    @classmethod
    def parse(cls, match, axis):
        return cls()


# a face is indexed like the cell to its right: R is that cell, L the one before it
class FaceSideOffset(CellOffset):
    pattern = r"(L+|R+)"

    @classmethod
    def parse(cls, match, axis):
        word = match.group(1)
        step = -len(word) if word[0] == "L" else len(word) - 1
        return cls(cls.unit(axis, step))

    @property
    def reach(self):
        # on a high interface the face's right cell is itself the first halo layer, hence s + 1
        s = sum(self.offset)
        return -s if s < 0 else s + 1


class CellFaceOffset(CellOffset):
    pattern = r"(low|high)"
    kind = FaceRange

    def __init__(self, high):
        super().__init__((0, 0, 0))
        self.high = high

    @classmethod
    def parse(cls, match, axis):
        return cls(match.group(1) == "high")

    # a face array has one plane more than the cells along its axis, so a cell's high face is an interior face
    @property
    def reach(self):
        return 0

    def binding(self, member):
        di, dj, dk = self.unit(member.spec.axis) if self.high else (0, 0, 0)
        return f"{member.cname}.at<{di}, {dj}, {dk}>({member.shape}, at_)"


class HaloOffset(ZeroOffset):
    pattern = r"halo"


class ListOffset(BaseOffset):
    pattern = r"list"

    def binding(self, member):
        return f"{member.cname}.list({member.shape}, at_)"


class BroadcastOffset(BaseOffset):
    pattern = r"broadcast"
    kind = Broadcast

    def binding(self, member):
        return f"{member.cname}.whole({member.shape})"


class BaseCarriedOffset(BaseOffset):
    carried = True

    @staticmethod
    def inward(intface):
        return CellOffset.unit(intface.axis, 1 if intface.is_low else -1)

    def binding(self, member):
        return f"{member.cname}.atoffset<{self.key}>({member.shape}, at_)"


class InteriorOffset(BaseCarriedOffset):
    pattern = r"interior(\d*)"

    def __init__(self, n=0):
        self.n = n

    @classmethod
    def parse(cls, match, axis):
        return cls(int(match.group(1) or 0))

    @property
    def key(self):
        return f"interior{self.n or ''}"

    def displace(self, intface, layer):
        di, dj, dk = self.inward(intface)
        n = layer + 1 + self.n
        return (di * n, dj * n, dk * n)


class MirrorOffset(BaseCarriedOffset):
    pattern = r"mirror(\d*)"

    def __init__(self, n=0):
        self.n = n

    @classmethod
    def parse(cls, match, axis):
        return cls(int(match.group(1) or 0))

    @property
    def key(self):
        return f"mirror{self.n or ''}"

    def displace(self, intface, layer):
        di, dj, dk = self.inward(intface)
        cells = CellCenterRange(intface.blk).exts[intface.axis]
        # clamped to the block's deepest cell: a slab thinner than the halo reflects its last cell into every layer
        n = layer + 1 + min(layer + self.n, cells - 1)
        return (di * n, dj * n, dk * n)


class PlaneOffset(BaseCarriedOffset):
    pattern = r"plane"
    kind = InterfacePlaneRange

    @property
    def key(self):
        return "plane"

    def displace(self, intface, layer):
        blk, axis = intface.blk, intface.axis
        ng, cells = blk.ng, blk.extents[axis] - 1
        # a plane array has no halo, so the offset drops ng in-plane and lands the layer's halo cell on the plane's one index
        normal = ng - 1 - layer if intface.is_low else ng + cells + layer
        return tuple(-normal if m == axis else -ng for m in range(3))


class IntfaceOffset(BaseCarriedOffset):
    pattern = r"intface"
    kind = FaceRange

    def __init__(self, axis):
        self.axis = axis

    @classmethod
    def parse(cls, match, axis):
        return cls(axis)

    @property
    def key(self):
        return "intface"

    def displace(self, intface, layer):
        di, dj, dk = self.inward(intface)
        # the interface face is indexed like the first interior cell on a low interface, the first halo cell on a high one
        n = layer + (1 if intface.is_low else 0)
        return (di * n, dj * n, dk * n)


###############################################################################
# Execution policies
###############################################################################
class BasePolicy:
    # the word the tag's over= names
    name = None
    # the C++ policy whose decode turns a tile's item into the thread's item
    cxx = None
    # the body's parameters
    signature = ""
    # which of the backend's tile knobs sizes a tile of these items
    tilekind = "cells"
    # the C++ index per item; None for a flat kernel whose item is one element, not one cell
    index = "pg::Index"
    # the offsets admitted after 'at'
    offsets = ()

    def __init__(self, axis=None):
        self.axis = axis
        # the carried offsets the kernel names, in the order the range carries them
        self.carried = []

    @classmethod
    def named(cls, name, axis=None):
        return subclass_where(cls, name=name)(axis)

    def offset(self, text):
        text = " ".join(text.split())
        if not text:
            return ZeroOffset()
        for offset in self.offsets:
            match = re.fullmatch(offset.pattern, text)
            if match:
                return offset.parse(match, self.axis)
        raise ValueError(f"a kernel over {self.name} has no offset {text!r}")

    def expects(self, offset):
        return (offset.kind or CellCenterRange, offset.axis)

    def blkrange(self, blk):
        return CellCenterRange(blk)

    def ncomps(self, kernel):
        return 1

    def runs(self, kernel):
        return ["return true;"]


class CellPolicy(BasePolicy):
    name = "cells"
    cxx = "pg::CellPolicy"
    offsets = (CellOffset, CellFaceOffset, ListOffset)


class ListPolicy(CellPolicy):
    name = "listed"
    cxx = "pg::ListPolicy"

    def listing(self, kernel):
        # the list is the one integer name at list; a float name at list is read flat, a row per slot
        listed = [b for b in kernel.bindings if isinstance(b.offset, ListOffset) and b.member.spec.dtype is not None]
        if len(listed) != 1:
            raise ValueError(f"{kernel} runs over a list: declare one integer name at list, not {len(listed)}")
        return listed[0]

    def runs(self, kernel):
        listed = self.listing(kernel)
        return [f"return {listed.offset.binding(listed.member)}.locate(at_);"]

    def blkrange(self, blk):
        return SlotRange(blk)


class FacePolicy(BasePolicy):
    name = "faces"
    offsets = (FaceSideOffset, BroadcastOffset, InteriorOffset)

    def __init__(self, axis=None):
        super().__init__(axis)
        if axis is None:
            raise ValueError("a kernel over faces takes an axis")

    @property
    def cxx(self):
        return f"pg::FacePolicy<{self.axis}>"

    def expects(self, offset):
        if isinstance(offset, (FaceSideOffset, InteriorOffset)):
            return (CellCenterRange, None)
        if offset.kind is not None:
            return (offset.kind, None)
        return (FaceRange, self.axis)

    # a carried offset reads inward from the face, and only an interface knows which way inward is
    def blkrange(self, blk):
        if self.carried:
            raise ValueError(f"a kernel over {self.name} at {self.carried[0].key} runs over interfaces, not a range of a block")
        return FaceRange(blk, self.axis)

    def haloranges(self, intface):
        # the interface face's own layer: the first interior cell (-1) on a low interface, the first halo cell on a high one
        layer = -1 if intface.is_low else 0
        displaced = [o.displace(intface, layer) for o in self.carried]
        return [(start, extent, displaced) for start, extent in FaceRange(intface.blk, self.axis).plane(intface)]


class HaloCellPolicy(CellPolicy):
    name = "halocells"
    offsets = (
        HaloOffset,
        InteriorOffset,
        MirrorOffset,
        IntfaceOffset,
        PlaneOffset,
        BroadcastOffset,
    )

    def blkrange(self, blk):
        raise ValueError(f"a kernel over {self.name} runs over interfaces, not a range of a block")

    def haloranges(self, intface):
        cells = CellCenterRange(intface.blk)
        ranges = []
        for layer in range(intface.blk.ng):
            displaced = [o.displace(intface, layer) for o in self.carried]
            for start, extent in cells.halolayer(intface, layer):
                ranges.append((start, extent, displaced))
        return ranges


class SlabPolicy(CellPolicy):
    name = "slab"
    offsets = (BroadcastOffset,)

    def expects(self, offset):
        return (offset.kind or BaseRange, None)


class ElementPolicy(BasePolicy):
    name = "elements"
    cxx = "pg::ElementPolicy"
    signature = "const int i"
    tilekind = "elements"
    index = None

    def expects(self, offset):
        return (BaseRange, None)

    def blkrange(self, blk):
        return SlotRange(blk)

    def ncomps(self, kernel):
        views = (m for m in kernel.members if isinstance(m, ViewMember))
        return prod(next(views).spec.comps)
