import ctypes

import numpy as np

from peregrine.config import ConfigError
from peregrine.misc import get_local_rank
from peregrine.backend.base.backend import BaseBackend
from peregrine.backend.device.graph import DeviceGraph


class DeviceFunction:
    # the most threads a block takes
    threads = 1024

    def __init__(self, backend, function, kernel):
        self.backend, self.function = backend, function
        self.reduce = kernel.reduce
        # the blocks' partials of a reduction, grown to the largest launch, and their address as an argument
        self.partials = self.holder = self.npartials = None

    # the driver copies the values at launch or at node add, so the addresses need not stay valid after
    def arguments(self, functor, tiling):
        addresses = [ctypes.addressof(functor), ctypes.addressof(tiling.abi)]
        if self.reduce:
            if self.npartials is None or self.npartials < tiling.tiles:
                # flat, a block's values together, whatever the backend's layout
                self.partials = self.backend.alloc((tiling.tiles * self.reduce.count,), name="partials")
                self.npartials = tiling.tiles
                self.holder = ctypes.c_void_p(self.partials.ptr)
            addresses.append(ctypes.addressof(self.holder))
        return (ctypes.c_void_p * len(addresses))(*addresses)

    def __call__(self, functor, tiling, out=None):
        block = min(tiling.tile, self.threads)
        args = self.arguments(functor, tiling)
        self.backend.driver.launch(self.function, tiling.tiles, block, args, self.backend.kernels)
        if self.reduce:
            count = self.reduce.count
            self.backend.kernels.synchronize()
            partials = self.partials.get()[: tiling.tiles * count].reshape(tiling.tiles, count)
            out[...] = self.reduce.ufunc.reduce(partials, axis=0)

    def node(self, graph, deps, functor, tiling):
        if self.reduce:
            raise ValueError(f"{self.function}: a reduction's values are read on the host; it is launched, never a node")
        block = min(tiling.tile, self.threads)
        return graph.add_kernel(self.function, tiling.tiles, block, self.arguments(functor, tiling), deps)


class BaseDeviceBackend(BaseBackend):
    graphcls = DeviceGraph
    entries = f"{__package__}.kernels"
    fnqual = "__device__ inline"
    # non-const and external: const folds under nvcc and clang, static folds under clang
    memqual = "__constant__"
    # column-major: a trailing component is one contiguous run, so it crosses the bus alone
    order = "F"
    drivercls = None
    compilercls = None
    # the driver, started once a process, per leaf
    driver = None

    def __init__(self, cfg=None):
        super().__init__(cfg)
        # the launch bound the kernels are compiled with, (threads, waves); none for a backend with no case
        self.bounds = None
        if cfg is not None:
            threads = self.positive("launch-threads")
            waves = self.positive("launch-waves")
            self.bounds = (threads, waves)
            if self.tile("cells") > threads:
                raise ConfigError(f"backend-{self.name}.tile-cells", "is at most launch-threads.")
        leaf = type(self)
        if leaf.driver is None:
            leaf.driver = self.drivercls(get_local_rank())
        self.compiler = self.compilercls(self.driver.arch)
        self.kernels = self.driver.stream()
        self.copies = self.driver.stream()
        # the modules loaded, by the build they came from
        self.modules = {}

    def stop(self):
        self.driver.synchronize()

    # the entry is named for the kernel, so a profiler sees each by name
    def function(self, path, kernel):
        if path not in self.modules:
            self.modules[path] = self.driver.module(path.read_bytes())
        return DeviceFunction(self, self.modules[path].function(f"pg_{kernel.symbol}"), kernel)

    def memory(self, array):
        return self.driver.alloc(array.nbytes)

    def release(self, array):
        self.driver.free(array.ptr)

    def to_host(self, array, component):
        self.kernels.synchronize()
        self.copies.synchronize()
        if component is None:
            host = np.empty(array.shape, array.dtype, order=self.order)
            self.driver.memcpy(host.ctypes.data, array.ptr, array.nbytes)
            return host
        host = np.empty(array.shape[:-1], array.dtype, order="F")
        self.driver.memcpy(host.ctypes.data, array.ptr + component * host.nbytes, host.nbytes)
        return host

    def head(self, array, count):
        self.kernels.synchronize()
        self.copies.synchronize()
        host = np.empty(count, array.dtype)
        self.driver.memcpy(host.ctypes.data, array.ptr, host.nbytes)
        return host

    def async_pull(self, array, host):
        self.copies_await_kernels()
        self.driver.memcpy(host.ctypes.data, array.ptr, array.nbytes, self.copies)

    def async_push(self, array, host):
        self.driver.memcpy(array.ptr, host.ctypes.data, array.nbytes, self.copies)

    def await_copies(self):
        self.copies.synchronize()

    def copies_await_kernels(self):
        event = self.driver.event()
        event.record(self.kernels)
        self.copies.wait(event)

    def kernels_await_copies(self):
        event = self.driver.event()
        event.record(self.copies)
        self.kernels.wait(event)

    def pinned(self, shape, dtype=None):
        dtype = np.dtype(self.fpdtype if dtype is None else dtype)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        ptr = self.driver.pinned(nbytes)
        buf = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_byte))
        array = np.ctypeslib.as_array(buf, (nbytes,)).view(dtype).reshape(shape, order=self.order)
        array[...] = 0
        return array

    # on the kernels' stream, so a kernel reading the array runs after the copy
    def from_host(self, array, values, wait):
        host = np.require(values, array.dtype, self.order)
        self.driver.memcpy(array.ptr, host.ctypes.data, array.nbytes, self.kernels)
        if wait:
            self.kernels.synchronize()
