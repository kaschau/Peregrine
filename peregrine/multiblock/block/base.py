from functools import cached_property

from peregrine.multiblock.interface import BaseInterface


class BaseBlock:
    # the arrays a cut, a merge or a reorient carries with the block; the rest are derived again
    carried = ()

    def __init__(self, nblki):
        self.nblki = nblki

        # base block in uncut grid
        self.origblk = nblki
        # slice of base block in uncut grid (None = entire block)
        self.baseslice = None

        self.ni = 0
        self.nj = 0
        self.nk = 0

    @cached_property
    def intfaces(self):
        return [self._new_intface(nface) for nface in range(1, 7)]

    @property
    def basebounds(self):
        whole = (0, self.ni - 1, 0, self.nj - 1, 0, self.nk - 1)
        return self.baseslice or whole

    @property
    def ncells(self):
        return (self.ni - 1) * (self.nj - 1) * (self.nk - 1)

    def set_extents(self, ni, nj, nk):
        self.ni, self.nj, self.nk = int(ni), int(nj), int(nk)

    # taken before either half is resized: set_extents allocates the arrays anew
    def split_along(self, axis, cutidx):
        low, high = [slice(None)] * 3, [slice(None)] * 3
        low[axis] = slice(0, cutidx + 1)
        high[axis] = slice(cutidx, None)
        halves = {}
        for name in self.carried:
            values = getattr(self, name).get()
            halves[name] = (values[tuple(low)], values[tuple(high)])
        return halves

    def _new_intface(self, nface):
        # the kind of interface this kind of block is bounded by
        return BaseInterface(nface)

    def get_intface(self, nface):
        assert 1 <= nface <= 6, "nface must be between (1,6)"
        return self.intfaces[int(nface) - 1]

    # an interface adjacent to the connection keeps its end; only the connection itself swaps ends (its nbrnface)
    def nbrnfaceof(self, nface, *, across):
        interface, conn = self.get_intface(nface), self.get_intface(across)
        assert (
            interface.axis != conn.axis
        ), f"interface {nface} is not adjacent to the connection on interface {across}"
        axis, flip = interface.signedaxis(conn.orient[interface.axis])
        return 2 * axis + (1 if interface.is_low != flip else 2)
