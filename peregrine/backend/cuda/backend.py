from peregrine.backend.device.backend import BaseDeviceBackend
from peregrine.backend.cuda.compiler import CudaCompiler
from peregrine.backend.cuda.driver import CudaDriver


class CudaBackend(BaseDeviceBackend):
    name = "cuda"
    templates = f"{__package__}.kernels"
    drivercls = CudaDriver
    compilercls = CudaCompiler
