from peregrine.backend.device.backend import BaseDeviceBackend
from peregrine.backend.hip.compiler import HipCompiler
from peregrine.backend.hip.driver import HipDriver


class HipBackend(BaseDeviceBackend):
    name = "hip"
    templates = f"{__package__}.kernels"
    drivercls = HipDriver
    compilercls = HipCompiler
