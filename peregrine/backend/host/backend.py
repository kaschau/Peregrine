import ctypes

import numpy as np

from peregrine.backend.base.backend import BaseBackend


class HostFunction:
    def __init__(self, entry, reduce):
        self.entry = entry
        self.entry.argtypes = [ctypes.c_void_p] * (3 if reduce else 2)
        self.entry.restype = None

    def __call__(self, functor, tiling, out=None):
        args = [ctypes.addressof(functor), ctypes.addressof(tiling.abi)]
        if out is not None:
            args.append(out.ctypes.data)
        self.entry(*args)


class BaseHostBackend(BaseBackend):
    fnqual = "inline"
    # a const table the host compiler folds or loads as it chooses
    memqual = "static const"
    order = "C"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        # the kernel libraries loaded, by the build they came from
        self.libs = {}

    # loaded local so one unit's symbols never resolve to another's
    def function(self, path, kernel):
        if path not in self.libs:
            self.libs[path] = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
        return HostFunction(getattr(self.libs[path], f"pg_{kernel.symbol}"), kernel.reduce)

    def memory(self, array):
        array.data = np.zeros(array.shape, array.dtype, order=self.order)
        return array.data.ctypes.data

    # built on the owner's memory: numpy takes no strided window as a buffer
    def pooled(self, array, within, offset):
        shape, dtype = array.shape, array.dtype
        itemsize = dtype.itemsize
        start = within.ptr - within.owner.ptr + offset * itemsize
        strides = [s * itemsize for s in array.strides]
        array.data = np.ndarray(shape, dtype, within.owner.data, start, strides)
        return array.data.ctypes.data

    def release(self, array):
        pass

    def to_host(self, array, component):
        if component is None:
            return array.data.copy(order=self.order)
        return np.array(array.data[..., component], order=self.order)

    def view(self, array):
        return array.data

    def head(self, array, count):
        return array.data.reshape(-1, order=self.order)[:count].copy()

    def async_pull(self, array, host):
        host[...] = array.data

    def async_push(self, array, host):
        array.data[...] = host

    # a host copy or launch is done when it returns, so nothing waits
    def await_copies(self):
        pass

    def copies_await_kernels(self):
        pass

    def kernels_await_copies(self):
        pass

    def pinned(self, shape, dtype=None):
        return np.zeros(shape, self.fpdtype if dtype is None else dtype, order=self.order)

    def from_host(self, array, values, wait):
        array.data[...] = values
