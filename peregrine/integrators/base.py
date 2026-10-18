from collections import namedtuple

from peregrine.launch.kernel import Kernel

# one stage reads the solution from Q, writes the derivative into dQ, leaves the solution in Qout
StageRegisters = namedtuple("StageRegisters", "Q dQ Qout")


class BaseIntegrator:
    # what the config calls it
    name = None
    # the registers of the solution it needs, the solution itself counted
    nregs = 1

    def __init__(self, controller):
        self.controller = controller

    ###########################################################################
    # What it needs
    ###########################################################################
    def decl_kerns(self, provider):
        self.provider = provider
        self.kernels = {}
        return self.kernels

    # when step, dt multiplies the last pair's term, so callers put dQ last
    def combine(self, out, *pairs, step=True):
        weights = {}
        for register, weight in pairs:
            weights[register] = weights.get(register, 0.0) + weight
        own = weights.pop(out, 0.0)
        reads = [register for register in weights if weights[register] != 0.0]
        options = dict(terms=len(reads), own=own != 0.0, step=step)
        xs = {f"X{n}": r for n, r in enumerate(reads)}
        ws = {f"w{n}": weights[r] for n, r in enumerate(reads)}
        arguments = {"out": out, **xs, **ws}
        if own:
            arguments["wout"] = own
        if step:
            arguments["dt"] = "dt"
        template = f"{__package__}.kernels.combine"
        return Kernel(self.provider, template, options=options, **arguments)

    # register 0 holds the solution between steps
    def stageregs(self, registers):
        raise NotImplementedError

    ###########################################################################
    # Stepping
    ###########################################################################
    def advance_graph(self, g, regs, n):
        raise NotImplementedError

    def step(self, solver, dt):
        raise NotImplementedError

    ###########################################################################
    # What it says
    ###########################################################################
    def report(self):
        controller = self.controller.name
        return f"  Time Integrator: {self.name}\n  Step Size: {controller}\n"
