from types import SimpleNamespace

import numpy as np

import peregrine as pg

from tests.gases import configure, primitives
from tests.riemann import RiemannProblem

Ru = 8314.46261815324
# constants only, nothing from the library: one species, and two of different weight
air = {"Air": {"MW": 28.97, "cp0": 1000.0}}
binary = {"N2": {"MW": 28.0134, "cp0": 1040.0}, "O2": {"MW": 31.9988, "cp0": 918.0}}


def gasconstant(gas):
    (species,) = gas.values()
    return Ru / species["MW"]


def gamma(gas):
    (species,) = gas.values()
    return species["cp0"] / (species["cp0"] - gasconstant(gas))


# the compositions the config can name today, by what a test calls them
schemes = {
    "kepaec": dict(primary="kepaec", recon="constant"),
    "kepaec4": dict(primary="kepaec4", recon="constant"),
    "rusanov": dict(primary="rusanov", recon="constant"),
    "ausmup": dict(primary="ausmup", recon="constant"),
    "hllc": dict(primary="hllc", recon="constant"),
    "muscl-koren-rusanov": dict(primary="rusanov", recon="muscl", limiter="koren"),
    "muscl-koren-ausmup": dict(primary="ausmup", recon="muscl", limiter="koren"),
    "muscl-koren-hllc": dict(primary="hllc", recon="muscl", limiter="koren"),
    "kepaec+rusanov by jameson": dict(
        formulation="blended",
        primary="kepaec",
        recon="constant",
        secondary="rusanov",
        switch="jameson",
        values={"p-gain": 5.0, "gamma-gain": 1.0},
    ),
}


# no face normal is an axis, so every area vector has three components
class JostledCube(pg.mesher.CubeMesher):
    def shape_blk(self, blk, i, j, k):
        super().shape_blk(blk, i, j, k)
        nodes = blk.nodes.get()
        inner = np.s_[1:-1, 1:-1, 1:-1]
        spacing = min(L / (n - 1) for L, n in zip(self.lengths, self.blkdims))
        rng = np.random.default_rng(7)
        nodes[inner] += rng.uniform(-1, 1, nodes[inner].shape) * 0.15 * spacing
        blk.nodes.set(nodes)


def case(backend, dims, lengths, periodic, gas=air, origin=(0.0, 0.0, 0.0), integrator="tvd-rk3", mesher=pg.mesher.CubeMesher, cls=pg.multiblock.Solver, **adv):
    config = pg.Config()
    config["simulation"]["simulator"] = "euler"
    config["mixture"]["species"] = gas
    config["adv-flux"].update(adv)
    config["time-integration"]["integrator"] = integrator
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    mesh = mesher(
        mbdims=[1, 1, 1],
        blkdims=list(dims),
        lengths=list(lengths),
        origin=list(origin),
        periodic=list(periodic),
        bcnames=dict.fromkeys(range(1, 7), "walls"),
    )
    return pg.get_solver(config, mesh, backend(config), cls=cls)


def line(backend, nx, periodic=True, length=1.0, origin=0.0, axis=0, **kw):
    dims, lengths, periodics = [2, 2, 2], [0.1, 0.1, 0.1], [False, False, False]
    dims[axis], lengths[axis], periodics[axis] = nx, length, periodic
    origins = [0.0, 0.0, 0.0]
    origins[axis] = origin
    return case(backend, dims, lengths, periodics, origin=origins, **kw)


def box(backend, dims=(7, 6, 5), gas=binary, **kw):
    return case(backend, dims, [1.0, 1.0, 1.0], [True] * 3, gas=gas, mesher=JostledCube, **kw)


def interior(blk):
    ng = blk.ng
    return np.s_[ng:-ng, ng:-ng, ng:-ng]


def faces(mb, axis, reach=1):
    blk = mb.blocks[0]
    ng = blk.ng
    name = "ijk"[axis]
    arrays = dict(Q=blk.Q.get(), state=blk.state.get())
    arrays["S"], arrays["F"] = getattr(blk, f"{name}S").get(), getattr(blk, f"{name}F").get()
    # the other two axes over the interior proper, so a periodic halo cell is read but never a halo edge
    arrays = {k: np.moveaxis(a, axis, 0)[:, ng:-ng, ng:-ng] for k, a in arrays.items()}
    n = arrays["Q"].shape[0] - 2 * ng
    cells = {o: {k: arrays[k][ng + o : ng + n + 1 + o] for k in ("Q", "state")} for o in range(-reach, reach)}
    return SimpleNamespace(
        S=arrays["S"][ng : ng + n + 1],
        F=arrays["F"][ng : ng + n + 1],
        Q={o: c["Q"] for o, c in cells.items()},
        state={o: c["state"] for o, c in cells.items()},
    )


def stepregs(mb):
    return mb.integ.stageregs(mb.regs)[0]


###############################################################################
# The density wave: an entropy wave advected once around a periodic line
###############################################################################
def wave_errors(backend, nx, axis=0, u=100.0, gas=air, **adv):
    mb = line(backend, nx, axis=axis, gas=gas, **adv)
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    xc = blk.xcc.get()[..., axis]
    rho = 1.2 + 0.6 * np.sin(2 * np.pi * xc)
    q[..., 0] = 101325.0
    q[..., 1 + axis] = u
    q[..., 4] = q[..., 0] / (rho * gasconstant(gas))
    mb.set_prims([q])
    cmax = np.sqrt(gamma(gas) * q[..., 0].max() / rho.min())
    dx = 1.0 / (nx - 1)
    n = int(round(1.0 / (abs(u) * 0.1 * dx / (abs(u) + cmax))))
    dt = 1.0 / (abs(u) * n)
    for _ in range(n):
        mb.step(dt)
    along = [blk.ng] * 3
    along[axis] = interior(blk)[axis]
    error = np.abs(blk.Q.get()[tuple(along)][:, 0] - rho[tuple(along)])
    return error.max(), error.mean()


###############################################################################
# Toro's shock tubes on a walled line
###############################################################################
def toro(backend, case_number, length=1.0, cells_per_unit=200, gas=air, **adv):
    test = RiemannProblem.toro(case_number, gamma=gamma(gas), R=gasconstant(gas))
    nx = int(round(length * cells_per_unit)) + 1
    mb = line(backend, nx, periodic=False, length=length, origin=test.x0 - 0.5 * length, gas=gas, **adv)
    blk = mb.blocks[0]
    inner = interior(blk)
    xc = blk.xcc.get()[..., 0]
    q = primitives(mb, blk)
    left = xc <= test.x0
    q[..., 0] = np.where(left, test.pL, test.pR)
    q[..., 1] = np.where(left, test.uL, test.uR)
    q[..., 4] = np.where(left, test.TL, test.TR)
    mb.set_prims([q])
    started = blk.Q.get()[inner].sum(axis=(0, 1, 2))
    for _ in range(int(round(test.t / test.dt))):
        mb.step(test.dt)
    Q = blk.Q.get()[inner]
    fields = mb.sim.statefields
    p = blk.state.get()[inner][..., fields.index("p")][:, 0, 0]
    x = xc[inner][:, 0, 0]
    out = SimpleNamespace(test=test, finite=bool(np.isfinite(Q).all()))
    if not out.finite:
        return out
    moved = Q.sum(axis=(0, 1, 2))[[0, 4]] / started[[0, 4]] - 1.0
    out.mass, out.energy = abs(moved[0]), abs(moved[1])
    # the fan's heads: a shock's speed or a rarefaction's head either way
    pM, uM = test.star()
    g1, g2 = (test.gamma - 1.0) / (2.0 * test.gamma), (test.gamma + 1.0) / (2.0 * test.gamma)
    sL = test.uL - test.cL * (np.sqrt(g2 * pM / test.pL + g1) if pM > test.pL else 1.0)
    sR = test.uR + test.cR * (np.sqrt(g2 * pM / test.pR + g1) if pM > test.pR else 1.0)
    head = (test.x0 + sL * test.t, test.x0 + sR * test.t)
    assert x[0] < head[0] and head[1] < x[-1], f"the fan {head} reaches a wall of {(x[0], x[-1])}"
    # the fan widened a tenth for the smearing, within the line
    lo = max(head[0] - 0.1 * (head[1] - head[0]), x[0])
    hi = min(head[1] + 0.1 * (head[1] - head[0]), x[-1])
    # a wall the fluid moves at sends a wave no faster than |u| + c below Mach 4; one it rests at sends none
    reach = (x[0] + (abs(test.uL) + test.cL) * test.t * bool(test.uL), x[-1] - (abs(test.uR) + test.cR) * test.t * bool(test.uR))
    assert reach[0] <= lo and reach[1] >= hi, f"the walls' waves {reach} reach the fan {(lo, hi)}"
    fan = (x > lo) & (x < hi)
    exact = test.solve(x)
    rho, u = Q[:, 0, 0, 0], Q[:, 0, 0, 1] / Q[:, 0, 0, 0]
    out.rho_err = np.abs(rho - exact["rho"])[fan].mean() / np.abs(exact["rho"][fan]).mean()
    # the plateau between the contact and the right wave's near edge, its middle two fifths
    edge = sR if pM > test.pR else uM + test.cR * (pM / test.pR) ** g1
    a, b = test.x0 + uM * test.t, test.x0 + edge * test.t
    plateau = (x > a + 0.3 * (b - a)) & (x < b - 0.3 * (b - a))
    out.star, out.cells = (pM, uM), int(plateau.sum())
    out.p_err = np.abs(p[plateau] / pM - 1.0).max()
    # u relative to the problem's fastest velocity, since u* may be none
    out.u_err = np.abs(u[plateau] - uM).max() / max(abs(uM), abs(test.uL), abs(test.uR))
    return out


def sod(backend, **adv):
    return toro(backend, 0, **adv)


###############################################################################
# The gate's box, in a precision
###############################################################################
def gatebox(backend, gas="air", precision="double", dt=1e-9):
    config = pg.Config()
    configure(config, gas, "navier-stokes")
    config["simulation"]["precision"] = precision
    config["simulation"]["dt"] = dt
    config["bcvals"]["walls"] = {"kind": "no-slp-adia-wall"}
    mesh = pg.mesher.CubeMesher(
        mbdims=[2, 2, 1],
        blkdims=[8, 7, 6],
        lengths=[1, 1, 1],
        periodic=[True, False, True],
        bcnames={3: "walls", 4: "walls"},
    )
    mb = pg.get_solver(config, mesh, backend(config))
    perturb(mb)
    return mb


def perturb(mb):
    prims = []
    for blk in mb.blocks:
        rng = np.random.default_rng(blk.nblki)
        q = primitives(mb, blk)
        q[..., 0] *= 1 + 0.05 * rng.random(q.shape[:3])
        q[..., 1:4] = 30 * rng.random(q.shape[:3] + (3,))
        q[..., 4] *= 1 + 0.1 * rng.random(q.shape[:3])
        if mb.sim.ne > 5:
            base = rng.random(mb.sim.ne - 5) / (mb.sim.ne - 5)
            q[..., 5:] = base * (1 + 0.01 * rng.random(q.shape[:3] + (mb.sim.ne - 5,)))
        prims.append(q)
    mb.set_prims(prims)
