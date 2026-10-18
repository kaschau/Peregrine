from peregrine.backend.device.compiler import BaseDeviceCompiler
from peregrine.backend.hip.rtc import HipRtc


class HipCompiler(BaseDeviceCompiler):
    suffix = ".hsaco"
    rtccls = HipRtc
    fallback = "hipcc"

    def __init__(self, arch):
        super().__init__(arch)
        self.flags = [
            f"--offload-arch={self.arch}",
            "-std=c++20",
            "-O3",
            *(["-g"] if self.debug else ["-DNDEBUG"]),
            # LLVM hoists baked literals out of loops and spills them
            "-mllvm",
            "-disable-machine-licm",
        ]

    def command(self, source, out):
        cmd = ["hipcc", "--genco", "-x", "hip", f"--offload-arch={self.arch}", "-std=c++20", "-O3"]
        cmd += ["-g"] if self.debug else ["-DNDEBUG"]
        return cmd + ["-mllvm", "-disable-machine-licm", "-o", str(out), str(source)]
