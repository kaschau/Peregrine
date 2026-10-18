import numpy as np

from peregrine.launch.kernel import Kernel
from peregrine.plugins.simulator.base import BaseSimulatorPlugin


class ViscousSponge(BaseSimulatorPlugin):
    name = "viscsponge"
    simulators = "navier-stokes.*"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        origin = np.array(cfgsect["origin"], float)
        ending = np.array(cfgsect["ending"], float)
        self.length = np.linalg.norm(ending - origin)
        self.normal = (ending - origin) / self.length
        self.start = float(origin @ self.normal)
        self.mult = float(cfgsect["multiplier"])

    def launches(self, provider):
        nx, ny, nz = (float(n) for n in self.normal)
        numbers = dict(nx=nx, ny=ny, nz=nz, start=self.start, length=float(self.length), mult=self.mult)
        self.kernel = Kernel(provider, f"{__package__}.kernels.viscsponge", xcc="xcc", trstate="trstate", **numbers)
        # over the cells the transport kernel has just made: the interior and the first halo layer
        return {"prerhs": [(self.kernel, "shallow")]}
