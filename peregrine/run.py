from peregrine.integrators import get_integrator
from peregrine.multiblock import Solver
from peregrine.simulator import get_simulator


def get_solver(cfg, mesh, backend, state=None, cls=Solver):
    simulator, integrator = get_simulator(cfg), get_integrator(cfg)
    return cls(cfg, mesh, backend, simulator, integrator, state)
