class BaseController:
    # what the config calls it
    name = None

    def __init__(self, cfg):
        self.cfg = cfg

    def stepsize(self, solver):
        raise NotImplementedError


class Fixed(BaseController):
    name = "fixed"

    def stepsize(self, solver):
        return self.cfg["simulation"]["dt"]


class CFL(BaseController):
    name = "cfl"

    def stepsize(self, solver):
        sim = self.cfg["simulation"]
        dt = min(sim["cfl"] / solver.cflmax()["combined"], sim["dt-max"])
        # shortened to land on the next time a plugin acts at
        times = [p.tnext(solver) for p in solver.plugins.values()]
        now = solver.tme.get()
        return min([dt, *(t - now for t in times if t is not None)])
