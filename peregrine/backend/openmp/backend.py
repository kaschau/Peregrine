from peregrine.backend.host.backend import BaseHostBackend
from peregrine.backend.openmp.compiler import OpenMPCompiler


class OpenMPBackend(BaseHostBackend):
    name = "openmp"
    templates = f"{__package__}.kernels"
    entries = f"{__package__}.kernels"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is not None:
            self.compiler = OpenMPCompiler(self.cfgsect["cxx"])
