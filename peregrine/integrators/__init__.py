from peregrine.misc import subclass_where
from peregrine.integrators.base import BaseIntegrator
from peregrine.integrators.controllers import CFL, BaseController, Fixed
from peregrine.integrators.rungekutta import RK1, RK2, RK3, RK4, RK34, BaseRungeKutta

__all__ = [
    "BaseController",
    "BaseIntegrator",
    "CFL",
    "Fixed",
    "get_integrator",
    "BaseRungeKutta",
    "RK1",
    "RK2",
    "RK3",
    "RK34",
    "RK4",
]


def get_integrator(cfg):
    ctrl = subclass_where(BaseController, name=cfg["simulation"]["controller"])
    ti = cfg["time-integration"]
    integ = subclass_where(BaseIntegrator, name=ti["integrator"])
    return integ(ctrl(cfg))
