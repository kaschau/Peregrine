import numpy as np

import peregrine as pg
from peregrine.launch.kernel import Kernel
from peregrine.launch.ranges import CellCenterRange
from peregrine.simulator.mixture import Ru
from peregrine.simulator.mixture.eos import RealGasModel

from tests.gases import primitives


def config(species, eos, trange=(300.0, 3500.0)):
    config = pg.Config()
    config["mixture"]["species"] = species
    config["mixture"]["eos"] = eos
    config["mixture"]["T-range"] = trange
    config["simulation"]["simulator"] = "euler"
    config["bcvals"]["wall"] = {"kind": "slp-adia-wall"}
    return config


def mesh():
    return pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[2, 2, 2], lengths=[1, 1, 1], defaultbc="wall")


def solver(backend, species, eos, trange=(300.0, 3500.0), cls=pg.multiblock.Solver):
    cfg = config(species, eos, trange)
    return pg.get_solver(cfg, mesh(), backend(cfg), cls=cls)


class WithEosKernels(pg.multiblock.Solver):
    def _decl_arrays(self):
        super()._decl_arrays()
        self.decl_array("qj", CellCenterRange, comps=self.sim.ne - 3)
        self.decl_array("hi", CellCenterRange, comps=self.sim.mixture.ns)

    def _decl_kerns(self):
        super()._decl_kerns()
        self.kernels["drho"] = lambda Q: Kernel(self.provider, f"{__package__}.kernels.drho", Q=Q, state="state", qj="qj")
        self.kernels["hpartial"] = lambda Q: Kernel(self.provider, f"{__package__}.kernels.hpartial", Q=Q, state="state", hi="hi")


def composition(mb, **fractions):
    return np.array([fractions.get(name, 0.0) for name in mb.sim.mixture.spnames])


def setstate(mb, p, T, Y):
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    q[..., 0], q[..., 1:4], q[..., 4], q[..., 5:] = p, 0.0, T, np.asarray(Y)[:-1]
    mb.set_prims([q])


def cell(mb, name="state"):
    blk = mb.blocks[0]
    ng = blk.ng
    return getattr(blk, name).get()[ng, ng, ng]


def state(mb):
    values = dict(zip(mb.sim.statefields, cell(mb)))
    values["h"] = values["e"] + values["p"] / values["rho"]
    return values


# the state's temperature is the eos Newton's starting guess; below 1 K the kernel starts from 300 K instead
def set_tguess(mb, by=None, to=None):
    blk = mb.blocks[0]
    held = blk.state.get()
    slot = mb.sim.statefields.index("T")
    held[..., slot] = to if to is not None else held[..., slot] + by
    blk.state.set(held)


# the cubic in its textbook form, on PEREGRINE's own constants
cubic = RealGasModel.cubic


def cubicroots(A, B):
    u, w = cubic["u"], cubic["w"]
    roots = np.roots([1.0, -(1.0 + B - u * B), A + w * B**2 - u * B - u * B**2, -(A * B + w * B**2 + w * B**3)])
    return np.sort(roots[np.abs(roots.imag) < 1e-12].real)


def leastgibbs(Z, A, B):
    sq = np.sqrt(cubic["u"] ** 2 - 4.0 * cubic["w"])
    sigma, eps = 0.5 * (cubic["u"] + sq), 0.5 * (cubic["u"] - sq)
    Z = Z[Z > B]
    g = Z - 1.0 - np.log(Z - B) + A / (B * (sigma - eps)) * np.log((Z + eps * B) / (Z + sigma * B))
    return Z[np.argmin(g)]


def cubic_ab(sp, p, T, Y):
    MW = np.array([s["MW"] for s in sp])
    Tc = np.array([s["Tcrit"] for s in sp])
    pc = np.array([s["pcrit"] for s in sp])
    w = np.array([s["acentric"] for s in sp])
    X = Y / MW
    X /= X.sum()
    MWmix = 1.0 / np.sum(Y / MW)
    m = cubic["m"][0] + cubic["m"][1] * w + cubic["m"][2] * w**2
    alpha = (1.0 + m * (1.0 - np.sqrt(T / Tc))) ** 2
    a = cubic["omega_a"] * (Ru * Tc) ** 2 / pc * alpha
    b = cubic["omega_b"] * Ru * Tc / pc
    am = np.sum(np.outer(X, X) * np.sqrt(np.outer(a, a)))
    bm = np.sum(X * b)
    return am * p / (Ru * T) ** 2, bm * p / (Ru * T), MWmix


# subtracts no volume translation, so it is only right for species without a vshift
def textbookdensity(sp, p, T, Y):
    A, B, MWmix = cubic_ab(sp, p, T, Y)
    Z = leastgibbs(cubicroots(A, B), A, B)
    return p * MWmix / (Z * Ru * T)


def cubicvolume(p, T, Tc, pc, w, shift=0.0):
    sp = [dict(MW=1.0, Tcrit=Tc, pcrit=pc, acentric=w)]
    A, B, _ = cubic_ab(sp, p, T, np.array([1.0]))
    # the unpack takes the one real root a supercritical state has
    (Z,) = cubicroots(A, B)
    return Z * Ru * T / p - shift
