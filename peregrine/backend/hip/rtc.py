from ctypes import POINTER, c_char_p, c_int, c_size_t, c_void_p

from peregrine.backend.base.wrapper import BaseWrapper
from peregrine.backend.device.compiler import BaseRtc, CompilationError


class HiprtcError(Exception):
    pass


class HiprtcWrapper(BaseWrapper):
    libname = "hiprtc"
    statuses = {6: CompilationError, "*": HiprtcError}
    functions = [
        (c_int, "hiprtcCreateProgram", POINTER(c_void_p), c_char_p, c_char_p, c_int, POINTER(c_char_p), POINTER(c_char_p)),
        (c_int, "hiprtcDestroyProgram", POINTER(c_void_p)),
        (c_int, "hiprtcCompileProgram", c_void_p, c_int, POINTER(c_char_p)),
        (c_int, "hiprtcGetProgramLogSize", c_void_p, POINTER(c_size_t)),
        (c_int, "hiprtcGetProgramLog", c_void_p, c_char_p),
        (c_int, "hiprtcGetCodeSize", c_void_p, POINTER(c_size_t)),
        (c_int, "hiprtcGetCode", c_void_p, c_char_p),
    ]

    def bname(self, function):
        return function.removeprefix("hiprtc")


class HipRtc(BaseRtc):
    wrapper = HiprtcWrapper
    suffix = ".hip"
