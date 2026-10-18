import numpy as np

from peregrine.config import ConfigError
from peregrine.backend.abi import noffsets
from peregrine.backend.base.graph import Graph
from peregrine.backend.allocation import Allocation


class BaseBackend:
    # what the config and the command line call it, lower case; its config section is backend-<name>
    name = None
    # the package of what this backend writes itself into every unit: its head
    templates = None
    # the package of this backend's entry points, a launch's and a reduction's
    entries = None
    # the class of graph a recipe populates and the solver binds, run as this backend runs its graphs
    graphcls = Graph
    # how this backend qualifies a function a kernel calls, written before every such function
    fnqual = None
    # how it qualifies a constant kept in memory, written before its definition at namespace scope
    memqual = None
    # literal or memory, the config's when there is a case
    constform = "literal"
    # the layout arrays are kept in, C with the last index fastest or F with the first
    order = None

    def __init__(self, cfg=None):
        # the case's precision; double for a backend with no case, a grid's
        precision = cfg["simulation"]["precision"] if cfg else "double"
        dtypes = {"double": np.float64, "single": np.float32}
        self.fpdtype = np.dtype(dtypes[precision])
        # the compiler the kernels are built with; a backend with no case builds none
        self.compiler = None
        # the backend's section of the config; a backend with no case, a grid's, has none
        self.cfgsect = None if cfg is None else cfg[f"backend-{self.name}"]
        if self.cfgsect is not None:
            self.constform = self.cfgsect["constform"]

    def __repr__(self):
        return f"{type(self).__name__}({self.order!r})"

    def positive(self, key):
        value = self.cfgsect[key]
        if not isinstance(value, int) or value < 1:
            message = f"{value!r} is not a positive integer."
            raise ConfigError(f"backend-{self.name}.{key}", message)
        return value

    def tile(self, kind):
        return self.positive(f"tile-{kind}")

    def graph(self, name):
        return self.graphcls(name, self)

    def stop(self):
        pass

    # the callable takes the filled functor, the launch's tiling and a reduction's values array, None for a kernel writing arrays
    def function(self, path, kernel):
        raise NotImplementedError

    @property
    def fpctype(self):
        return np.ctypeslib.as_ctypes_type(self.fpdtype)

    def tplargs(self):
        fpdtype = "float" if self.fpdtype == np.float32 else "double"
        return dict(backend=self, fpdtype=fpdtype, finfo=np.finfo(self.fpdtype), fnqual=self.fnqual, memqual=self.memqual, constform=self.constform, noffsets=noffsets)

    def alloc(self, shape, dtype=None, name=None):
        return Allocation(shape, self, dtype, name=name)

    # zeroed memory of the array's size, as an address
    def memory(self, array):
        raise NotImplementedError

    def pooled(self, array, within, offset):
        return within.ptr + offset * array.dtype.itemsize

    def release(self, array):
        raise NotImplementedError

    def to_host(self, array, component):
        raise NotImplementedError

    def head(self, array, count):
        raise NotImplementedError

    # queued behind the kernels queued so far, so it reads what they leave; async_push is not
    def async_pull(self, array, host):
        raise NotImplementedError

    def async_push(self, array, host):
        raise NotImplementedError

    def await_copies(self):
        raise NotImplementedError

    def copies_await_kernels(self):
        raise NotImplementedError

    def kernels_await_copies(self):
        raise NotImplementedError

    # never released: taken once and kept for the run
    def pinned(self, shape, dtype=None):
        raise NotImplementedError

    def from_host(self, array, values, wait):
        raise NotImplementedError
