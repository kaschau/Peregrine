from ctypes import POINTER, c_char_p, c_int, c_size_t, c_void_p

from peregrine.backend.base.wrapper import BaseWrapper
from peregrine.backend.device.compiler import BaseRtc, CompilationError


class NvrtcError(Exception):
    pass


class NvrtcWrapper(BaseWrapper):
    libname = "nvrtc"
    statuses = {6: CompilationError, "*": NvrtcError}
    functions = [
        (c_int, "nvrtcCreateProgram", POINTER(c_void_p), c_char_p, c_char_p, c_int, POINTER(c_char_p), POINTER(c_char_p)),
        (c_int, "nvrtcDestroyProgram", POINTER(c_void_p)),
        (c_int, "nvrtcCompileProgram", c_void_p, c_int, POINTER(c_char_p)),
        (c_int, "nvrtcGetProgramLogSize", c_void_p, POINTER(c_size_t)),
        (c_int, "nvrtcGetProgramLog", c_void_p, c_char_p),
        (c_int, "nvrtcGetCUBINSize", c_void_p, POINTER(c_size_t)),
        (c_int, "nvrtcGetCUBIN", c_void_p, c_char_p),
    ]

    # the names BaseRtc calls, shared with HIPRTC: GetCUBIN is GetCode
    def bname(self, function):
        return function.removeprefix("nvrtc").replace("CUBIN", "Code")


class CudaRtc(BaseRtc):
    wrapper = NvrtcWrapper
    suffix = ".cu"
