from ctypes import c_void_p


class Stream:
    def __init__(self, driver):
        self.driver = driver
        self.ptr = driver._stream()

    def synchronize(self):
        self.driver._stream_synchronize(self.ptr)

    def wait(self, event):
        self.driver._stream_wait(self.ptr, event.ptr)


class Event:
    def __init__(self, driver):
        self.driver = driver
        self.ptr = driver._event()

    def record(self, stream):
        self.driver._event_record(self.ptr, stream.ptr)


class Module:
    def __init__(self, driver, code):
        self.driver = driver
        self.ptr = driver._module(code)

    def function(self, name):
        return self.driver._function(self.ptr, name)


class DriverGraph:
    def __init__(self, driver):
        self.driver = driver
        self.ptr = driver._graph()

    def add_kernel(self, function, grid, block, args, deps):
        ndeps = len(deps)
        deps = (c_void_p * ndeps)(*deps) if ndeps else None
        return self.driver._graph_kernel(self.ptr, function, grid, block, args, deps, ndeps)

    def instantiate(self):
        return ExecGraph(self.driver, self)


class ExecGraph:
    def __init__(self, driver, graph):
        self.driver, self.graph = driver, graph
        self.ptr = driver._instantiate(graph.ptr)

    def launch(self, stream):
        self.driver._graph_launch(self.ptr, stream.ptr)


class BaseDriver:
    # the wrapper of the vendor's library
    wrapper = None

    def __init__(self, localrank):
        self.lib = self.wrapper()
        # the device this rank takes, started, as the vendor handles it
        device = self._take_device(localrank % self._ndevices())
        # its architecture as the compiler spells it
        self.arch = self._arch(device)

    def stream(self):
        return Stream(self)

    def event(self):
        return Event(self)

    def module(self, code):
        return Module(self, code)

    def graph(self):
        return DriverGraph(self)

    def memcpy(self, dst, src, nbytes, stream=None):
        if stream is None:
            self._memcpy(dst, src, nbytes)
        else:
            self._memcpy_async(dst, src, nbytes, stream.ptr)

    def launch(self, function, grid, block, args, stream):
        self._launch(function, grid, block, args, stream.ptr)

    def alloc(self, nbytes):
        # a zero-byte allocation gives no address; an empty array still needs one
        nbytes = max(nbytes, 1)
        ptr = self._alloc(nbytes)
        # the clear runs on the null stream, which the backend's streams never wait on: so the host does
        self._memset(ptr, nbytes)
        self.synchronize()
        return ptr

    def pinned(self, nbytes):
        return self._pinned(max(nbytes, 1))

    def synchronize(self):
        raise NotImplementedError

    def free(self, ptr):
        raise NotImplementedError

    def _ndevices(self):
        raise NotImplementedError

    def _take_device(self, ordinal):
        raise NotImplementedError

    def _arch(self, device):
        raise NotImplementedError

    def _alloc(self, nbytes):
        raise NotImplementedError

    def _memset(self, ptr, nbytes):
        raise NotImplementedError

    def _pinned(self, nbytes):
        raise NotImplementedError

    def _memcpy(self, dst, src, nbytes):
        raise NotImplementedError

    def _memcpy_async(self, dst, src, nbytes, stream):
        raise NotImplementedError

    def _launch(self, function, grid, block, args, stream):
        raise NotImplementedError

    def _module(self, code):
        raise NotImplementedError

    def _function(self, module, name):
        raise NotImplementedError

    def _stream(self):
        raise NotImplementedError

    def _stream_synchronize(self, stream):
        raise NotImplementedError

    def _stream_wait(self, stream, event):
        raise NotImplementedError

    def _event(self):
        raise NotImplementedError

    def _event_record(self, event, stream):
        raise NotImplementedError

    def _graph(self):
        raise NotImplementedError

    def _graph_kernel(self, graph, function, grid, block, args, deps, ndeps):
        raise NotImplementedError

    def _instantiate(self, graph):
        raise NotImplementedError

    def _graph_launch(self, exec, stream):
        raise NotImplementedError
