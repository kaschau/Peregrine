import shutil
import warnings
from ctypes import c_char_p, c_size_t, c_void_p, create_string_buffer

from peregrine.backend.base.compiler import BaseCompiler


class CompilationError(Exception):
    pass


class BaseRtc:
    # the wrapper of the vendor's library
    wrapper = None
    # the suffix a source is named with
    suffix = None

    def __init__(self):
        self.lib = self.wrapper()

    def compile(self, name, text, flags):
        prog = c_void_p()
        self.lib.CreateProgram(prog, text.encode(), f"{name}{self.suffix}".encode(), 0, None, None)
        try:
            encoded = [f.encode() for f in flags]
            try:
                self.lib.CompileProgram(prog, len(flags), (c_char_p * len(flags))(*encoded))
            except CompilationError:
                size = c_size_t()
                self.lib.GetProgramLogSize(prog, size)
                log = create_string_buffer(size.value)
                self.lib.GetProgramLog(prog, log)
                raise CompilationError(log.value.decode()) from None
            size = c_size_t()
            self.lib.GetCodeSize(prog, size)
            code = create_string_buffer(size.value)
            self.lib.GetCode(prog, code)
        finally:
            self.lib.DestroyProgram(prog)
        return code.raw


class BaseDeviceCompiler(BaseCompiler):
    # the leaf's compiler library class, a BaseRtc
    rtccls = None
    # the compiler as a process, by name, on the path or not
    fallback = None

    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.rtc = self.rtccls()

    @property
    def key(self):
        return repr((type(self.rtc).__name__, self.flags))

    def compile(self, text, out):
        try:
            out.write_bytes(self.rtc.compile(out.stem, text, self.flags))
        except CompilationError as error:
            if shutil.which(self.fallback) is None:
                raise RuntimeError(f"{self.source(text, out)} did not compile:\n{error}") from None
            warnings.warn(f"{out.stem}: {type(self.rtc).__name__} refused the unit, {self.fallback} compiled it:\n{error}")
            self.run(self.command, text, out)

    def command(self, source, out):
        raise NotImplementedError
