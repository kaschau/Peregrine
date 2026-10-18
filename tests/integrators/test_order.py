import numpy as np
import pytest
from numpy.polynomial import Polynomial

import peregrine as pg
from peregrine.multiblock import Solver

from tests.integrators.cases import box, schemes

# the problem: dQ/dt = a Q, or dQ/dt = a cos(a t) Qref, over T, at two steps
a, T = 2.0, 0.5
steps = (8, 16)

# the order each scheme reaches on each forcing
order = {
    "linear": dict(zip(schemes, (1, 2, 3, 3, 4))),
    # a pure-time forcing is a quadrature of the stage times: Simpson's rule for the three-stage schemes and up
    "cosine": dict(zip(schemes, (1, 2, 4, 4, 4))),
}


class OdeSolver(Solver):
    forcing = "linear"

    def rhs(self, *, Q, dQ):
        t = self.tme.get()
        for blk in self.blocks:
            q = getattr(blk, Q).get()
            if self.forcing == "linear":
                d = a * q
            else:
                d = a * np.cos(a * t) * blk.Qref
            getattr(blk, dQ).set(d)


class Clock(Solver):
    def rhs(self, *, Q, dQ):
        self.times.append(self.tme.get())
        super().rhs(Q=Q, dQ=dQ)


def solver(backend, integrator, dt, cls):
    config, mesh = box(integrator=integrator, dims=(3, 3, 3), dt=dt)
    config["ics"]["u"] = 10.0
    return pg.get_solver(config, mesh, backend(config), cls=cls)


def exact(forcing, Q0, t):
    if forcing == "linear":
        return Q0 * np.exp(a * t)
    return Q0 * (1.0 + np.sin(a * t))


def error(backend, integrator, forcing, nsteps):
    dt = T / nsteps
    mb = solver(backend, integrator, dt, OdeSolver)
    mb.forcing = forcing
    blk = mb.blocks[0]
    blk.Qref = blk.Q.get()
    Q0 = blk.Q.get()
    for _ in range(nsteps):
        mb.step(dt)
    got = blk.Q.get()[blk.interior]
    want = exact(forcing, Q0, nsteps * dt)[blk.interior]
    scale = np.abs(want).max(axis=(0, 1, 2))
    rows = scale > 0
    err = np.abs(got - want).max(axis=(0, 1, 2))
    return (err[rows] / scale[rows]).max(), mb


def amplification(integ, regs):
    z = Polynomial([0.0, 1.0])
    held = {r: Polynomial([0.0]) for r in regs}
    held[regs[0]] = Polynomial([1.0])
    for n, stage in enumerate(integ.stageregs(regs)):
        # dQ/dt = lambda Q: the derivative register is the state's polynomial, lambda folded into z
        held[stage.dQ] = held[stage.Q]
        for kernel in integ.stagekerns(n, regs):
            binds, options = kernel.binds, kernel.options
            out = binds["out"]
            total = binds["wout"] * held[out] if options["own"] else Polynomial([0.0])
            for m in range(options["terms"]):
                term = held[binds[f"X{m}"]]
                if options["step"] and m == options["terms"] - 1:
                    term = z * term
                total = total + binds[f"w{m}"] * term
            held[out] = total
    return held[regs[0]]


def limits(R):
    t = np.arange(0.0, 10.0, 1e-3)

    def extent(direction):
        over = np.flatnonzero(np.abs(R(direction * t)) > 1.0 + 1e-9)
        return float(t[over[0] - 1]) if len(over) else float(t[-1])

    real, imaginary = extent(-1.0), extent(1.0j)
    # a scheme that leaves the imaginary axis at once reads a sliver, zeroed here
    return real, (imaginary if imaginary > 0.01 * real else 0.0)


@pytest.mark.parametrize("integrator", schemes)
def test_the_order_on_a_linear_equation(backend, integrator):
    # measured 0.92, 1.93, 2.93, 2.95, 3.93 between eight and sixteen steps
    e1, mb = error(backend, integrator, "linear", steps[0])
    e2, _ = error(backend, integrator, "linear", steps[1])
    p = order["linear"][integrator]
    assert np.log2(e1 / e2) >= p - 0.15
    # the error is the amplification polynomial's exactly: the combine kernels do its arithmetic, to rounding (measured 1.3e-15)
    R = amplification(mb.integ, mb.regs)
    z = a * T / steps[0]
    predicted = abs(R(z) ** steps[0] - np.exp(steps[0] * z)) / np.exp(steps[0] * z)
    assert abs(e1 - predicted) < 1e-12


@pytest.mark.parametrize("integrator", schemes)
def test_the_order_on_a_forcing_in_time(backend, integrator):
    # only the stage times are seen here; a tvd-rk3 with its second stage at a quarter measures 1.05, not 4 (measured 0.97, 2.00, 4.00, 4.00, 4.00)
    e1, _ = error(backend, integrator, "cosine", steps[0])
    e2, _ = error(backend, integrator, "cosine", steps[1])
    assert np.log2(e1 / e2) >= order["cosine"][integrator] - 0.15


@pytest.mark.parametrize(
    "integrator, fractions",
    [
        ("rk1", [0.0]),
        ("rk2", [0.0, 1.0]),
        ("tvd-rk3", [0.0, 1.0, 0.5]),
        ("rk34", [0.0, 0.5, 1.0, 0.5]),
        ("rk4", [0.0, 0.5, 0.5, 1.0]),
    ],
)
def test_every_right_hand_side_is_taken_at_its_stage_time(backend, integrator, fractions):
    dt = 1e-6
    mb = solver(backend, integrator, dt, Clock)
    mb.times = []
    start = mb.tme.get()
    mb.step(dt)
    assert mb.times == [start + f * dt for f in fractions]
    assert mb.tme.get() == start + dt


@pytest.mark.parametrize(
    "integrator, coefficients, real, imaginary",
    [
        ("rk1", [1, 1], 2.0, 0.0),
        ("rk2", [1, 1, 1 / 2], 2.0, 0.0),
        ("tvd-rk3", [1, 1, 1 / 2, 1 / 6], 2.512, 3**0.5),
        # the four-stage third-order scheme carries a fourth term that is not the exponential's
        ("rk34", [1, 1, 1 / 2, 1 / 6, 1 / 48], 5.149, 2.156),
        ("rk4", [1, 1, 1 / 2, 1 / 6, 1 / 24], 2.785, 2.828),
    ],
)
def test_the_amplification_is_the_exponential_to_the_order(backend, integrator, coefficients, real, imaginary):
    mb = solver(backend, integrator, 1e-6, Solver)
    R = amplification(mb.integ, mb.regs)
    assert np.allclose(R.coef, coefficients, rtol=0.0, atol=1e-15)
    assert len(R.coef) == len(coefficients)
    got = limits(R)
    assert abs(got[0] - real) < 0.01 and abs(got[1] - imaginary) < 0.01
