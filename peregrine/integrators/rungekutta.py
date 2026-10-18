from peregrine.integrators.base import BaseIntegrator, StageRegisters


class BaseRungeKutta(BaseIntegrator):
    nregs = 3
    # (fraction of dt, wQ0, wQ, wdQ) per stage: Qout = wQ Q + wQ0 Qstart + wdQ dt dQ
    stages = ()

    def stageregs(self, registers):
        start, stage, dQ = registers[0], registers[1], registers[-1]
        last = len(self.stages) - 1
        Q = [start if n == 0 else stage for n in range(len(self.stages))]
        Qout = [start if n == last else stage for n in range(len(self.stages))]
        return [StageRegisters(Q=q, dQ=dQ, Qout=o) for q, o in zip(Q, Qout)]

    def stagekerns(self, n, registers):
        _, wQ0, wQ, wdQ = self.stages[n]
        Q, dQ, Qout = self.stageregs(registers)[n]
        return [self.combine(Qout, (Q, wQ), (registers[0], wQ0), (dQ, wdQ))]

    def advance_graph(self, g, regs, n):
        after = ()
        for i, kernel in enumerate(self.stagekerns(n, regs)):
            # interior only: a halo is exchanged or set by a boundary before anything reads it
            g.add_launch(f"combine-{i}", kernel, "interior", after=after)
            after = (f"combine-{i}",)

    # an advance is at the next stage's time, since that stage's rhs reads what it produces
    def stagetimes(self, start, dt):
        fractions = [frac for frac, *_ in self.stages[1:]] + [1.0]
        return [start + frac * dt for frac in fractions]

    def step(self, solver, dt):
        times = self.stagetimes(solver.tme.get(), dt)
        regs = self.stageregs(solver.regs)
        for n, (t, stage) in enumerate(zip(times, regs)):
            solver.rhs(Q=stage.Q, dQ=stage.dQ)
            solver.tme.set(t)
            solver.advance(n)


class RK1(BaseRungeKutta):
    name = "rk1"
    nregs = 2
    stages = ((0.0, 0.0, 1.0, 1.0),)


class RK2(BaseRungeKutta):
    name = "rk2"
    stages = ((0.0, 0.0, 1.0, 1.0), (1.0, 0.5, 0.5, 0.5))


# Gottlieb and Shu, Total variation diminishing Runge-Kutta schemes, Math. Comp. 67(221), 1998
class RK3(BaseRungeKutta):
    name = "tvd-rk3"
    stages = (
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.75, 0.25, 0.25),
        (0.5, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
    )


class RK34(BaseRungeKutta):
    name = "rk34"
    stages = (
        (0.0, 0.0, 1.0, 0.5),
        (0.5, 0.0, 1.0, 0.5),
        (1.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
        (0.5, 0.0, 1.0, 0.5),
    )


class RK4(BaseRungeKutta):
    name = "rk4"
    nregs = 4
    # (fraction of dt, wdQ, wsum) per stage; the last has no wdQ of its own
    stages = (
        (0.0, 0.5, 1.0 / 6.0),
        (0.5, 0.5, 1.0 / 3.0),
        (0.5, 1.0, 1.0 / 3.0),
        (1.0, None, 1.0 / 6.0),
    )

    def stagekerns(self, n, registers):
        start, stage, total, dQ = registers
        _, wdQ, wsum = self.stages[n]
        # sum = sum + wsum dt dQ, the sum written rather than read at the first stage
        own = 0.0 if n == 0 else 1.0
        share = self.combine(total, (total, own), (dQ, wsum))
        if wdQ is None:
            # start = start + sum, the sum carrying the step already
            end = self.combine(start, (start, 1.0), (total, 1.0), step=False)
            return [share, end]
        # stage = start + wdQ dt dQ
        return [share, self.combine(stage, (start, 1.0), (dQ, wdQ))]
