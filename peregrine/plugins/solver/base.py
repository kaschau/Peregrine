from peregrine.plugins.base import BasePlugin
from peregrine.plugins.solver.cadence import Cadence


class BaseSolverPlugin(BasePlugin):
    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        self.cadence = Cadence(cfgsect)

    def arrays(self, solver):
        return {}

    def start(self, solver):
        pass

    def due(self, solver):
        return self.cadence.due(solver)

    def tnext(self, solver):
        return self.cadence.next(solver)

    def __call__(self, solver):
        pass

    def finalize(self, solver):
        pass
