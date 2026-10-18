import numpy as np
import pytest

import peregrine as pg
from peregrine.simulator.bcs import Expression

from tests.gases import configure, hcoeff

# every kind sits on the low j face, where x and z vary over the face and y does not
X = {
    "p": "101325.0 + 2000.0 * x - 500.0 * z",
    "pt": "111325.0 + 2000.0 * x - 500.0 * z",
    "u": "10.0 + 5.0 * z",
    "v": "sin(3.0 * x) - 2.0 * x * z",
    "w": "2.0 * t",
    "T": "300.0 + 40.0 * z**2 + max(x, 0.5)",
    "Tt": "330.0 + 40.0 * z**2 + max(x, 0.5)",
    "mflux": "20.0 + 5.0 * x * z",
    "H2": "0.1 + 0.05 * z",
    "O2": "0.5 - 0.1 * x",
}

# every kind that reads values: the names its entry gives, and its physics
KINDS = {
    "sup-in-fa": (("p", "u", "v", "w", "T", "H2", "O2"), "euler"),
    "sub-in-fvt": (("u", "v", "w", "T", "H2", "O2"), "euler"),
    "sub-in-ftptt": (("pt", "Tt", "H2", "O2"), "euler"),
    "sub-in-fmt": (("mflux", "T", "H2", "O2"), "euler"),
    "sub-out-fp": (("p",), "euler"),
    "slp-isot-wall": (("T",), "euler"),
    "mov-adia-wall": (("u", "v", "w"), "navier-stokes"),
    "mov-isot-wall": (("u", "v", "w", "T"), "navier-stokes"),
}


def numpy_eval(text, x, y, z, t):
    names = dict(
        x=x, y=y, z=z, t=t, sin=np.sin, cos=np.cos, tan=np.tan, exp=np.exp,
        log=np.log, sqrt=np.sqrt, tanh=np.tanh, abs=np.abs, min=np.minimum,
        max=np.maximum, pow=np.power,
    )
    return eval(text, {"__builtins__": {}}, names)


class Case:
    def __init__(self, backend, kind, t=1.5e-3):
        names, physics = KINDS[kind]
        config = pg.Config()
        configure(config, "CH4_O2", physics)
        config["bcvals"]["named"] = {"kind": kind, **{n: X[n] for n in names}}
        config["bcvals"]["wall"] = {"kind": "slp-adia-wall"}
        mesh = pg.mesher.CubeMesher(
            mbdims=[1, 1, 1], blkdims=[6, 4, 5], lengths=[1, 1, 1], bcnames={3: "named"}, defaultbc="wall"
        )
        self.mb = pg.get_solver(config, mesh, backend(config))
        self.blk = self.mb.blocks[0]
        self.face = self.blk.get_intface(3)
        self.t = t
        self.mb.tme.set(t)
        self.mb.apply_bcs("preadvflux")
        centers = self.blk.jxc.get()
        self.x, self.y, self.z = (self.face.faceplane(centers[..., c]) for c in range(3))
        fields = list(self.mb.sim.statefields)
        self.T, self.p = fields.index("T"), fields.index("p")
        self.gamma, self.c = fields.index("gamma"), fields.index("c")
        self.species = list(self.mb.sim.mixture.spnames)

    def value(self, name, t=None):
        return numpy_eval(X[name], self.x, self.y, self.z, self.t if t is None else t)

    def halos(self, array):
        ng = self.blk.ng
        return [h[ng:-ng, ng:-ng] for h in self.face.halo(array)]

    def first(self, array):
        ng = self.blk.ng
        return array[:, ng][ng:-ng, ng:-ng]

    def halo_values(self, name):
        Q, state = self.blk.Q.get(), self.blk.state.get()
        if name in ("p", "T"):
            return [h[..., {"p": self.p, "T": self.T}[name]] for h in self.halos(state)]
        if name in "uvw":
            c = 1 + "uvw".index(name)
            return [h[..., c] / h[..., 0] for h in self.halos(Q)]
        n = self.species.index(name)
        return [h[..., 5 + n] / h[..., 0] for h in self.halos(Q)]

    def check_imposed(self, name):
        for got in self.halo_values(name):
            assert np.allclose(got, self.value(name), rtol=1e-12, atol=1e-12)


def close(got, want):
    assert np.allclose(got, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


@pytest.mark.parametrize("kind", ["sup-in-fa", "sub-in-fvt"])
def test_an_inlet_imposes_its_expressions(backend, kind):
    case = Case(backend, kind)
    for name in KINDS[kind][0]:
        case.check_imposed(name)


def test_an_exit_imposes_its_pressure_expression(backend):
    Case(backend, "sub-out-fp").check_imposed("p")


def test_a_mass_flux_inlet_carries_its_expressions(backend):
    case = Case(backend, "sub-in-fmt")
    for name in ("T", "H2", "O2"):
        case.check_imposed(name)
    Q = case.blk.Q.get()
    rho0 = case.first(Q)[..., 0]
    for halo in case.halos(Q):
        rho = halo[..., 0]
        # the face's averages of density and velocity carry the mass flux with the interior at rest: (rho0 + rho) / 2 * v / 2 = mflux
        close(halo[..., 2] / rho, 4.0 * case.value("mflux") / (rho0 + rho))
        assert np.allclose(halo[..., [1, 3]], 0.0, atol=1e-12 * np.abs(halo[..., 2]).max())


# without the reference enthalpy, which cancels in the difference it is used in
def enthalpy(mixture, Y, T):
    consts = mixture.constants()
    if mixture.eos.name == "cpg":
        return (Y @ consts["cp0"]) * T
    coeff = hcoeff(Y @ consts["cppolyR"])
    u = np.log(T)
    h = np.zeros_like(T)
    for m in reversed(range(coeff.shape[-1])):
        h = h * u + coeff[..., m]
    return h * T


def test_a_stagnation_inlet_carries_its_expressions(backend):
    case = Case(backend, "sub-in-ftptt")
    for name in ("H2", "O2"):
        case.check_imposed(name)
    fields = list(case.mb.sim.statefields)
    state = case.first(case.blk.state.get())
    gamma, c, T0 = state[..., case.gamma], state[..., case.c], state[..., fields.index("T")]
    gm1 = gamma - 1.0
    # the entry's composition at every face center: what it names, the rest none, the last what they leave
    carried = np.stack([case.value(n) if n in X else np.zeros_like(c) for n in case.species[:-1]], axis=-1)
    Y = np.concatenate([carried, 1.0 - carried.sum(axis=-1, keepdims=True)], axis=-1)
    Tt = case.value("Tt")
    Ht = c * c / gm1 + enthalpy(case.mb.sim.mixture, Y, Tt) - enthalpy(case.mb.sim.mixture, Y, T0)
    # the kernel's isentropic algebra: the halo sound speed cb from the Riemann invariant and the entry's total enthalpy, one ratio per face center
    Jm = 2.0 * c / gm1
    aq, bq, cq = 1.0 + 2.0 / gm1, -2.0 * Jm, gm1 * (0.5 * Jm * Jm - Ht)
    t1, t2 = -bq / (2.0 * aq), np.sqrt(np.maximum(bq * bq - 4.0 * aq * cq, 0.0)) / (2.0 * aq)
    cb = np.maximum(t1 + t2, t1 - t2)
    isen = 1.0 + 0.5 * gm1 * ((2.0 * cb / gm1 - Jm) / cb) ** 2
    for T, p in zip(case.halo_values("T"), case.halo_values("p")):
        close(T * isen, Tt)
        close(p * isen ** (gamma / gm1), case.value("pt"))


def test_an_isothermal_wall_holds_its_expression(backend):
    case = Case(backend, "slp-isot-wall")
    for got in case.halo_values("T"):
        close(got, np.broadcast_to(case.value("T"), got.shape))


@pytest.mark.parametrize("kind", ["mov-adia-wall", "mov-isot-wall"])
def test_a_moving_wall_reflects_about_its_expressions(backend, kind):
    case = Case(backend, kind)
    if kind == "mov-isot-wall":
        for got in case.halo_values("T"):
            close(got, np.broadcast_to(case.value("T"), got.shape))
    Q = case.blk.Q.get()
    image = case.first(Q)
    uimage = image[..., 1:4] / image[..., :1]
    wall = np.stack([np.broadcast_to(case.value(n), case.x.shape) for n in "uvw"], axis=-1)
    for halo in case.halos(Q):
        close(halo[..., 1:4] / halo[..., :1], 2.0 * wall - uimage)


def test_the_clock_an_expression_reads_is_the_steps_end(backend):
    case = Case(backend, "sub-in-fvt", t=1.5e-3)
    dt = 1e-7
    case.mb.step(dt)
    case.mb.prerhs("Q")
    for got in case.halo_values("w"):
        assert np.allclose(got, 2.0 * (1.5e-3 + dt), rtol=1e-12, atol=0.0)


def test_what_a_kernel_cannot_evaluate_is_refused():
    for text in ("q + 1.0", "foo(x)", "x if y else z", "[x, y]", "x @ y"):
        with pytest.raises(ValueError):
            Expression(text)
