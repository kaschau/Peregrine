from peregrine.misc import subclass_where
from peregrine.simulator.advec import BaseAdvecSimulator
from peregrine.simulator.advecdiff import BaseAdvecDiffSimulator
from peregrine.simulator.base import BaseSimulator
from peregrine.simulator.euler import EulerSimulator
from peregrine.simulator.eulerdf import EulerDoubleFluxSimulator
from peregrine.simulator.iovars import IoVars
from peregrine.simulator.navstokes import NavierStokesSimulator
from peregrine.simulator.navstokesdf import NavierStokesDoubleFluxSimulator

__all__ = [
    "BaseAdvecDiffSimulator",
    "BaseAdvecSimulator",
    "BaseSimulator",
    "EulerDoubleFluxSimulator",
    "EulerSimulator",
    "NavierStokesDoubleFluxSimulator",
    "IoVars",
    "NavierStokesSimulator",
    "get_simulator",
]


def get_simulator(cfg):
    name = cfg["simulation"]["simulator"]
    simulator = subclass_where(BaseSimulator, name=name)(cfg)
    simulator.validate()
    return simulator
