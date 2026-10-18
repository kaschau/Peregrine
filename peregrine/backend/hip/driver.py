import ctypes
from ctypes import POINTER, Structure, c_char_p, c_int, c_size_t, c_uint, c_ulonglong, c_void_p, create_string_buffer

from peregrine.backend.base.wrapper import BaseWrapper
from peregrine.backend.device.driver import BaseDriver


class HipError(Exception):
    pass


class HipInvalidValue(HipError):
    pass


class HipOutOfMemory(HipError):
    pass


# hipKernelNodeParams lays its fields out alphabetically, unlike CUDA's
class KernelNodeParams(Structure):
    _fields_ = [
        ("block", c_uint * 3),
        ("extra", POINTER(c_void_p)),
        ("func", c_void_p),
        ("grid", c_uint * 3),
        ("params", POINTER(c_void_p)),
        ("shared", c_uint),
    ]


class HipWrapper(BaseWrapper):
    libname = "amdhip64"
    statuses = {1: HipInvalidValue, 2: HipOutOfMemory, "*": HipError}
    # a copy whose direction the runtime tells from the addresses
    MEMCPY_DEFAULT = 4
    # a stream that never waits on the null stream, and an event that keeps no time
    STREAM_NON_BLOCKING = 1
    EVENT_DISABLE_TIMING = 2
    functions = [
        (c_int, "hipGetDeviceCount", POINTER(c_int)),
        (c_int, "hipSetDevice", c_int),
        (c_int, "hipDeviceSynchronize"),
        # ROCm 6 exports the properties call under its ABI suffix; the bare symbol fills the old layout, not the one _arch reads at its offset
        (c_int, "hipGetDevicePropertiesR0600", c_void_p, c_int),
        (c_int, "hipMalloc", POINTER(c_void_p), c_size_t),
        (c_int, "hipFree", c_void_p),
        (c_int, "hipMemset", c_void_p, c_int, c_size_t),
        (c_int, "hipHostMalloc", POINTER(c_void_p), c_size_t, c_uint),
        (c_int, "hipMemcpy", c_void_p, c_void_p, c_size_t, c_int),
        (c_int, "hipMemcpyAsync", c_void_p, c_void_p, c_size_t, c_int, c_void_p),
        (c_int, "hipStreamCreateWithFlags", POINTER(c_void_p), c_uint),
        (c_int, "hipStreamSynchronize", c_void_p),
        (c_int, "hipStreamWaitEvent", c_void_p, c_void_p, c_uint),
        (c_int, "hipEventCreateWithFlags", POINTER(c_void_p), c_uint),
        (c_int, "hipEventRecord", c_void_p, c_void_p),
        (c_int, "hipModuleLoadData", POINTER(c_void_p), c_void_p),
        (c_int, "hipModuleGetFunction", POINTER(c_void_p), c_void_p, c_char_p),
        (
            c_int,
            "hipModuleLaunchKernel",
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
        (c_int, "hipGraphCreate", POINTER(c_void_p), c_uint),
        (c_int, "hipGraphAddKernelNode", POINTER(c_void_p), c_void_p, POINTER(c_void_p), c_size_t, POINTER(KernelNodeParams)),
        (c_int, "hipGraphInstantiateWithFlags", POINTER(c_void_p), c_void_p, c_ulonglong),
        (c_int, "hipGraphLaunch", c_void_p, c_void_p),
    ]

    def bname(self, function):
        return function.removesuffix("R0600")


class HipDriver(BaseDriver):
    wrapper = HipWrapper

    def _ndevices(self):
        count = c_int()
        self.lib.hipGetDeviceCount(count)
        return count.value

    def _take_device(self, ordinal):
        self.lib.hipSetDevice(ordinal)
        return ordinal

    def _arch(self, device):
        # the device's properties as ROCm lays them out: its architecture by name
        props = create_string_buffer(2048)
        self.lib.hipGetDeviceProperties(props, device)
        return ctypes.cast(props[1160:], c_char_p).value.decode().split(":")[0]

    def synchronize(self):
        self.lib.hipDeviceSynchronize()

    def free(self, ptr):
        self.lib.hipFree(ptr)

    def _alloc(self, nbytes):
        ptr = c_void_p()
        self.lib.hipMalloc(ptr, nbytes)
        return ptr.value

    def _memset(self, ptr, nbytes):
        self.lib.hipMemset(ptr, 0, nbytes)

    def _pinned(self, nbytes):
        ptr = c_void_p()
        self.lib.hipHostMalloc(ptr, nbytes, 0)
        return ptr.value

    def _memcpy(self, dst, src, nbytes):
        self.lib.hipMemcpy(dst, src, nbytes, self.lib.MEMCPY_DEFAULT)

    def _memcpy_async(self, dst, src, nbytes, stream):
        self.lib.hipMemcpyAsync(dst, src, nbytes, self.lib.MEMCPY_DEFAULT, stream)

    def _launch(self, function, grid, block, args, stream):
        self.lib.hipModuleLaunchKernel(function, grid, 1, 1, block, 1, 1, 0, stream, args, None)

    def _module(self, code):
        ptr = c_void_p()
        self.lib.hipModuleLoadData(ptr, code)
        return ptr.value

    def _function(self, module, name):
        ptr = c_void_p()
        self.lib.hipModuleGetFunction(ptr, module, name.encode())
        return ptr.value

    def _stream(self):
        ptr = c_void_p()
        self.lib.hipStreamCreateWithFlags(ptr, self.lib.STREAM_NON_BLOCKING)
        return ptr.value

    def _stream_synchronize(self, stream):
        self.lib.hipStreamSynchronize(stream)

    def _stream_wait(self, stream, event):
        self.lib.hipStreamWaitEvent(stream, event, 0)

    def _event(self):
        ptr = c_void_p()
        self.lib.hipEventCreateWithFlags(ptr, self.lib.EVENT_DISABLE_TIMING)
        return ptr.value

    def _event_record(self, event, stream):
        self.lib.hipEventRecord(event, stream)

    def _graph(self):
        ptr = c_void_p()
        self.lib.hipGraphCreate(ptr, 0)
        return ptr.value

    def _graph_kernel(self, graph, function, grid, block, args, deps, ndeps):
        params = KernelNodeParams((block, 1, 1), None, function, (grid, 1, 1), args, 0)
        node = c_void_p()
        self.lib.hipGraphAddKernelNode(node, graph, deps, ndeps, params)
        return node.value

    def _instantiate(self, graph):
        ptr = c_void_p()
        self.lib.hipGraphInstantiateWithFlags(ptr, graph, 0)
        return ptr.value

    def _graph_launch(self, exec, stream):
        self.lib.hipGraphLaunch(exec, stream)
