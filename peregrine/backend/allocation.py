import ctypes

import numpy as np

from peregrine.backend import abi


class Allocation:
    def __init__(self, shape, backend, dtype=None, *, name=None, strides=None):
        self.shape = tuple(int(n) for n in shape)
        self.backend = backend
        self.dtype = np.dtype(backend.fpdtype if dtype is None else dtype)
        self.order = backend.order
        self.name = name
        # strides in elements: the layout's unless given, as a subview's are
        if strides is None:
            strides = self.cstrides(self.shape, self.order)
        self.strides = tuple(int(s) for s in strides)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        # the memory, as an address, and whose it is
        self.ptr = self._memory()
        # what a kernel receives, padded to the ABI's five axes
        ndim = len(self.shape)
        pad = [0] * (5 - ndim)
        extent = (ctypes.c_int * 5)(*self.shape, *pad)
        stride = (ctypes.c_long * 5)(*self.strides, *pad)
        self.info = abi.ArrInfo(self.ptr, ndim, extent, stride)

    # contiguous strides
    @staticmethod
    def cstrides(shape, order):
        out, step = [0] * len(shape), 1
        axes = range(len(shape)) if order == "F" else reversed(range(len(shape)))
        for axis in axes:
            out[axis] = step
            step *= shape[axis]
        return out

    def _memory(self):
        self.owner = self
        return self.backend.memory(self)

    def __del__(self):
        if getattr(self, "ptr", None) and getattr(self, "owner", None) is self:
            self.backend.release(self)

    def __repr__(self):
        kind = type(self).__name__
        return f"<{kind} {self.name or ''} {self.shape} on {self.backend!r}>"

    def get(self, component=None):
        return self.backend.to_host(self, component)

    # the memory itself, unsynchronised: for a reader with nothing in flight; a backend off the host has none
    def view(self):
        return self.backend.view(self)

    # the first count elements flat in memory order, not along the first axis
    def head(self, count):
        return self.backend.head(self, count)

    # queued behind the kernels queued so far; complete once the backend's await_copies returns
    def async_pull(self, host):
        assert host.shape == self.shape and host.dtype == self.dtype
        self.backend.async_pull(self, host)

    # a kernel reading it launches after the backend's kernels_await_copies
    def async_push(self, host):
        assert host.shape == self.shape and host.dtype == self.dtype
        self.backend.async_push(self, host)

    # with wait=False the values have to outlive the copy
    def set(self, values, wait=True):
        values = np.asarray(values, self.dtype)
        assert values.shape == self.shape, (values.shape, self.shape)
        self.backend.from_host(self, values, wait)


# the value lives on the host in double whatever the case's precision, so a clock stepped many times loses no time
class Scalar(Allocation):
    def __init__(self, backend, name, value=0.0):
        super().__init__((1,), backend, name=name)
        self.value = 0.0
        self.host = np.zeros(1, self.dtype)
        self.set(value)

    def set(self, value):
        self.value = float(value)
        self.host[0] = self.value
        super().set(self.host, wait=False)

    def get(self):
        return self.value


class Subview(Allocation):
    def __init__(self, array, offset, shape, strides=None, *, name=None):
        # the base takes its memory in its constructor, through _memory, which reads these
        self.array, self.offset = array, int(offset)
        backend, dtype = array.backend, array.dtype
        super().__init__(shape, backend, dtype, name=name, strides=strides)
        # every element of it inside the memory's owner, from where this window starts in it
        start = (self.ptr - self.owner.ptr) // self.dtype.itemsize
        spans = [s * (n - 1) for n, s in zip(self.shape, self.strides)]
        first = start + sum(min(0, d) for d in spans)
        last = start + sum(max(0, d) for d in spans)
        window = (self.shape, self.strides, start, self.owner.shape)
        assert 0 <= first and last < int(np.prod(self.owner.shape)), window

    # axes are 1-based and signed: window axis d runs along array axis |axes[d]| - 1, backwards where negative
    @classmethod
    def along(cls, array, origin, axes, extents, *, name=None):
        S = array.strides
        naxes = len(origin)
        # each axis of the window inside the array's, which the flat check of the two ends cannot see past the fastest axis
        for a, extent in zip(axes, extents):
            axis, sign = abs(a) - 1, (-1 if a < 0 else 1)
            ends = (origin[axis], origin[axis] + sign * (extent - 1)) if extent else ()
            if any(not 0 <= end < array.shape[axis] for end in ends):
                raise ValueError(f"a window on {array} runs along axis {axis} from {ends[0]} to {ends[1]}, past its {array.shape[axis]}")
        offset = sum(o * s for o, s in zip(origin, S))
        strides = [(-1 if a < 0 else 1) * S[abs(a) - 1] for a in axes]
        strides += S[naxes:]
        shape = tuple(extents) + array.shape[naxes:]
        return cls(array, offset, shape, strides, name=name)

    def _memory(self):
        self.owner = self.array.owner
        return self.backend.pooled(self, self.array, self.offset)

    def get(self, component=None):
        itemsize = self.dtype.itemsize
        host = self.owner.get()
        start = self.ptr - self.owner.ptr
        strides = [s * itemsize for s in self.strides]
        view = np.ndarray(self.shape, self.dtype, host, start, strides)
        return (view if component is None else view[..., component]).copy()
