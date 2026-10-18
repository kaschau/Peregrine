import numpy as np


class Cadence:
    def __init__(self, cfgsect):
        self.niterout = cfgsect.get("niter-out")
        self.dtout = cfgsect.get("dt-out")
        if self.niterout is None and self.dtout is None:
            self.niterout = 1

    def nreached(self, tme):
        # a step landed on a multiple of dtout can fall a few ulps short of it; the tolerance counts that as reached
        tol = 8.0 * np.spacing(max(abs(tme), self.dtout))
        return int((tme + tol) / self.dtout)

    def next(self, solver):
        if self.dtout is None:
            return None
        return (self.nreached(solver.tme.get()) + 1) * self.dtout

    def due(self, solver):
        if self.dtout is None:
            return solver.nrt % self.niterout == 0
        tme, dt = solver.tme.get(), solver.dt.get()
        return self.nreached(tme) != self.nreached(tme - dt)
