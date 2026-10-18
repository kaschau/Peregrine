from ctypes import POINTER, Structure, c_char_p, c_int, c_size_t, c_ubyte, c_uint, c_ulonglong, c_void_p

from peregrine.backend.base.wrapper import BaseWrapper
from peregrine.backend.device.driver import BaseDriver


class CudaError(Exception):
    pass


class CudaOutOfMemory(CudaError):
    pass


class CudaInvalidValue(CudaError):
    pass


class CudaNoDevice(CudaError):
    pass


class CudaIllegalAddress(CudaError):
    pass


class CudaLaunchFailed(CudaError):
    pass


class KernelNodeParams(Structure):
    _fields_ = [
        ("func", c_void_p),
        ("grid", c_uint * 3),
        ("block", c_uint * 3),
        ("shared", c_uint),
        ("params", POINTER(c_void_p)),
        ("extra", POINTER(c_void_p)),
    ]


class CudaWrapper(BaseWrapper):
    libname = "cuda"
    statuses = {
        1: CudaInvalidValue,
        2: CudaOutOfMemory,
        100: CudaNoDevice,
        700: CudaIllegalAddress,
        719: CudaLaunchFailed,
        "*": CudaError,
    }
    # device attributes
    COMPUTE_CAPABILITY_MAJOR = 75
    COMPUTE_CAPABILITY_MINOR = 76
    # a stream that never waits on the null stream, and an event that keeps no time
    STREAM_NON_BLOCKING = 1
    EVENT_DISABLE_TIMING = 2
    functions = [
        (c_int, "cuInit", c_uint),
        (c_int, "cuDeviceGetCount", POINTER(c_int)),
        (c_int, "cuDeviceGet", POINTER(c_int), c_int),
        (c_int, "cuDeviceGetAttribute", POINTER(c_int), c_int, c_int),
        (c_int, "cuDevicePrimaryCtxRetain", POINTER(c_void_p), c_int),
        (c_int, "cuCtxSetCurrent", c_void_p),
        (c_int, "cuCtxSynchronize"),
        (c_int, "cuMemAlloc_v2", POINTER(c_void_p), c_size_t),
        (c_int, "cuMemFree_v2", c_void_p),
        (c_int, "cuMemAllocHost_v2", POINTER(c_void_p), c_size_t),
        (c_int, "cuMemcpy", c_void_p, c_void_p, c_size_t),
        (c_int, "cuMemcpyAsync", c_void_p, c_void_p, c_size_t, c_void_p),
        (c_int, "cuMemsetD8_v2", c_void_p, c_ubyte, c_size_t),
        (c_int, "cuStreamCreate", POINTER(c_void_p), c_uint),
        (c_int, "cuStreamSynchronize", c_void_p),
        (c_int, "cuStreamWaitEvent", c_void_p, c_void_p, c_uint),
        (c_int, "cuEventCreate", POINTER(c_void_p), c_uint),
        (c_int, "cuEventRecord", c_void_p, c_void_p),
        (c_int, "cuModuleLoadDataEx", POINTER(c_void_p), c_void_p, c_uint, POINTER(c_int), POINTER(c_void_p)),
        (c_int, "cuModuleGetFunction", POINTER(c_void_p), c_void_p, c_char_p),
        (
            c_int,
            "cuLaunchKernel",
            c_void_p,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_void_p,
            POINTER(c_void_p),
            c_void_p,
        ),
        (c_int, "cuGraphCreate", POINTER(c_void_p), c_uint),
        (c_int, "cuGraphAddKernelNode", POINTER(c_void_p), c_void_p, POINTER(c_void_p), c_size_t, POINTER(KernelNodeParams)),
        (c_int, "cuGraphInstantiateWithFlags", POINTER(c_void_p), c_void_p, c_ulonglong),
        (c_int, "cuGraphLaunch", c_void_p, c_void_p),
    ]

    def __init__(self):
        super().__init__()
        # the driver is initialized before it is asked anything
        self.cuInit(0)

    def bname(self, function):
        return function.removesuffix("_v2")


class CudaDriver(BaseDriver):
    wrapper = CudaWrapper

    def _ndevices(self):
        count = c_int()
        self.lib.cuDeviceGetCount(count)
        return count.value

    def _take_device(self, ordinal):
        device = c_int()
        self.lib.cuDeviceGet(device, ordinal)
        # the device's primary context, current on this thread
        ctx = c_void_p()
        self.lib.cuDevicePrimaryCtxRetain(ctx, device)
        self.lib.cuCtxSetCurrent(ctx)
        return device.value

    def _arch(self, device):
        major, minor = c_int(), c_int()
        self.lib.cuDeviceGetAttribute(major, self.lib.COMPUTE_CAPABILITY_MAJOR, device)
        self.lib.cuDeviceGetAttribute(minor, self.lib.COMPUTE_CAPABILITY_MINOR, device)
        return f"sm_{major.value}{minor.value}"

    def synchronize(self):
        self.lib.cuCtxSynchronize()

    def free(self, ptr):
        self.lib.cuMemFree(ptr)

    def _alloc(self, nbytes):
        ptr = c_void_p()
        self.lib.cuMemAlloc(ptr, nbytes)
        return ptr.value

    def _memset(self, ptr, nbytes):
        self.lib.cuMemsetD8(ptr, 0, nbytes)

    def _pinned(self, nbytes):
        ptr = c_void_p()
        self.lib.cuMemAllocHost(ptr, nbytes)
        return ptr.value

    def _memcpy(self, dst, src, nbytes):
        self.lib.cuMemcpy(dst, src, nbytes)

    def _memcpy_async(self, dst, src, nbytes, stream):
        self.lib.cuMemcpyAsync(dst, src, nbytes, stream)

    def _launch(self, function, grid, block, args, stream):
        self.lib.cuLaunchKernel(function, grid, 1, 1, block, 1, 1, 0, stream, args, None)

    def _module(self, code):
        ptr = c_void_p()
        self.lib.cuModuleLoadDataEx(ptr, code, 0, None, None)
        return ptr.value

    def _function(self, module, name):
        ptr = c_void_p()
        self.lib.cuModuleGetFunction(ptr, module, name.encode())
        return ptr.value

    def _stream(self):
        ptr = c_void_p()
        self.lib.cuStreamCreate(ptr, self.lib.STREAM_NON_BLOCKING)
        return ptr.value

    def _stream_synchronize(self, stream):
        self.lib.cuStreamSynchronize(stream)

    def _stream_wait(self, stream, event):
        self.lib.cuStreamWaitEvent(stream, event, 0)

    def _event(self):
        ptr = c_void_p()
        self.lib.cuEventCreate(ptr, self.lib.EVENT_DISABLE_TIMING)
        return ptr.value

    def _event_record(self, event, stream):
        self.lib.cuEventRecord(event, stream)

    def _graph(self):
        ptr = c_void_p()
        self.lib.cuGraphCreate(ptr, 0)
        return ptr.value

    def _graph_kernel(self, graph, function, grid, block, args, deps, ndeps):
        params = KernelNodeParams(function, (grid, 1, 1), (block, 1, 1), 0, args, None)
        node = c_void_p()
        self.lib.cuGraphAddKernelNode(node, graph, deps, ndeps, params)
        return node.value

    def _instantiate(self, graph):
        ptr = c_void_p()
        self.lib.cuGraphInstantiateWithFlags(ptr, graph, 0)
        return ptr.value

    def _graph_launch(self, exec, stream):
        self.lib.cuGraphLaunch(exec, stream)
