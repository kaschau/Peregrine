import ctypes
import ctypes.util
import os
import sys
from pathlib import Path


class BaseWrapper:
    # the library's name, lib<name>.so or .dylib
    libname = None
    # the exception per status code, "*" for any other
    statuses = {}
    # the type a function returns a status as
    status = ctypes.c_int
    functions = []
    weak = []

    def __init__(self):
        self.lib = self.load()
        for restype, name, *argtypes in self.functions:
            self.wrap(name, restype, argtypes)
        for restype, name, *argtypes in self.weak:
            try:
                self.wrap(name, restype, argtypes)
            except AttributeError:
                pass

    def load(self):
        env = f"PEREGRINE_{self.libname.upper()}_LIBRARY_PATH"
        if path := os.environ.get(env):
            return ctypes.CDLL(path)
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        filename = f"lib{self.libname}{suffix}"
        dirs = [d for d in os.environ.get("PEREGRINE_LIBRARY_PATH", "").split(":") if d]
        for d in dirs:
            try:
                return ctypes.CDLL(str(Path(d, filename).absolute()))
            except OSError:
                pass
        return ctypes.CDLL(filename)

    # bound name
    def bname(self, function):
        return function

    def wrap(self, name, restype, argtypes):
        fn = getattr(self.lib, name)
        fn.restype, fn.argtypes = restype, argtypes
        if self.statuses and restype == self.status:
            fn.errcheck = self.check
        setattr(self, self.bname(name), fn)

    def check(self, status, fn, args):
        if status != 0:
            error = self.statuses.get(status, self.statuses.get("*"))
            raise error(f"{fn.__name__}: status {status}")
        return status
