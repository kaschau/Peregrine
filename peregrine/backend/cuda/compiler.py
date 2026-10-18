from peregrine.backend.cuda.rtc import CudaRtc
from peregrine.backend.device.compiler import BaseDeviceCompiler


class CudaCompiler(BaseDeviceCompiler):
    suffix = ".cubin"
    rtccls = CudaRtc
    fallback = "nvcc"

    def __init__(self, arch):
        super().__init__(arch)
        self.flags = [
            f"--gpu-architecture={self.arch}",
            "-std=c++20",
            # a unit is device code throughout: what carries no annotation is the device's
            "-default-device",
            *(["-G", "-lineinfo"] if self.debug else ["-DNDEBUG"]),
            "--fmad=true",
        ]

    def command(self, source, out):
        cmd = ["nvcc", "-cubin", "-x", "cu", f"-arch={self.arch}", "-std=c++20", "-O3"]
        cmd += ["-G", "-lineinfo"] if self.debug else ["-DNDEBUG"]
        return cmd + ["--fmad=true", "--Werror", "cross-execution-space-call", "-o", str(out), str(source)]
