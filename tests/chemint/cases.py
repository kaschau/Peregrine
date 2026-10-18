from pathlib import Path

import cantera as ct
import numpy as np

import peregrine as pg
from peregrine.launch.kernel import Kernel
from peregrine.launch.ranges import CellCenterRange

from tests.gases import primitives

ct.add_directory(str(Path(__file__).parent / "../../peregrine/simulator/mixture/database/mechanisms"))
mechanism = "CH4_O2_FFCMY.yaml"
# the tests' own production kernel on the package's rate macros, binding the solution, the state and omega
rates = f"{__package__}.kernels.production"


class WithRates(pg.multiblock.Solver):
    def _decl_arrays(self):
        super()._decl_arrays()
        self.decl_array("omega", CellCenterRange, comps=self.sim.mixture.ns)

    def _decl_kerns(self):
        super()._decl_kerns()
        self.kernels["rates"] = lambda Q: Kernel(self.provider, rates, Q=Q, state="state", omega="omega")


def config(mechanism, kind, precision="double", dt=1e-9, maxsubsteps=200, simulator="euler", trange=(300.0, 3500.0), **chemistry):
    config = pg.Config()
    sim = config["simulation"]
    sim["simulator"], sim["precision"], sim["dt"] = simulator, precision, dt
    mixture = config["mixture"]
    mixture["species"], mixture["eos"], mixture["T-range"] = mechanism, "tpg", trange
    if simulator != "euler":
        mixture["trans"] = "kinetic-theory"
    mixture["chemistry"] = "finite-rate" if kind else None
    config["chem-integration"]["kind"] = kind
    config["chem-integration"]["max-substeps"] = maxsubsteps
    for key, value in chemistry.items():
        config["chem-integration"][key.replace("_", "-")] = value
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    return config


def cell(backend, mechanism, kind, p, T, Y, dt=1e-9, maxsubsteps=200, precision="double", cls=WithRates, stepset=True, trange=(300.0, 3500.0), **chemistry):
    mesh = pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[2, 2, 2], lengths=[0.01, 0.01, 0.01], defaultbc="walls")
    cfg = config(mechanism, kind, precision, dt, maxsubsteps, trange=trange, **chemistry)
    mb = pg.get_solver(cfg, mesh, backend(cfg), cls=cls)
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    q[..., 0], q[..., 1:4], q[..., 4], q[..., 5:] = p, 0.0, T, np.asarray(Y)[:-1]
    mb.set_prims([q])
    if stepset:
        mb.dt.set(dt)
    mb.rhs(Q=mb.regs[0], dQ=mb.regs[-1])
    if kind is not None:
        mb.launch(mb.kernels["rates"](Q=mb.regs[0]), "interior")
    return mb


def duct(backend, kind="sub-stepped", nx=8, dt=1e-6, hot=2500.0, cold=400.0, cls=pg.multiblock.Solver, simulator="euler", **chemistry):
    # wide cells keep the acoustic CFL small at a step the chemistry is stiff over
    mesh = pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[nx + 1, 3, 3], lengths=[0.4, 0.1, 0.1], defaultbc="walls")
    cfg = config(mechanism, kind, dt=dt, simulator=simulator, **chemistry)
    mb = pg.get_solver(cfg, mesh, backend(cfg), cls=cls)
    blk = mb.blocks[0]
    ng = mb.ng
    q = primitives(mb, blk)
    T = np.full(q.shape[0], cold)
    T[: ng + nx // 2] = np.max(hot)
    T[ng : ng + nx // 2] = hot
    q[..., 0], q[..., 1:4], q[..., 4] = 10e5, 0.0, T[:, None, None]
    q[..., 5:] = stoichiometric()[:-1]
    mb.set_prims([q])
    mb.dt.set(dt)
    mb.rhs(Q=mb.regs[0], dQ=mb.regs[-1])
    return mb


def at_cell(mb, name):
    ng = mb.ng
    return getattr(mb.blocks[0], name).get()[ng, ng, ng]


def interior(mb, name):
    blk = mb.blocks[0]
    return getattr(blk, name).get()[blk.interior]


def dQ(mb):
    return at_cell(mb, mb.regs[-1])


def temperature(mb):
    return at_cell(mb, "state")[mb.sim.statefields.index("T")]


def ownstate(mb):
    Q = at_cell(mb, "Q")
    Y = Q[5:] / Q[0]
    return Q[0], np.append(Y, 1.0 - Y.sum()), temperature(mb)


# the list's layout: the total, a count per bin, then each bin's cell codes i + n0 (j + n1 k) in a block-sized slot
def listed(mb):
    blk = mb.blocks[0]
    bins, ng = mb.cfg["chem-integration"]["stiff-bins"], mb.ng
    flat = blk.stiffcells.get().reshape(-1, order=mb.backend.order)
    n0, n1, n2 = blk.stiffcells.shape[:3]
    total, counts = int(flat[0]), flat[1 : 1 + bins].astype(int)
    cells = []
    for b, count in enumerate(counts):
        start = b * n0 * n1 * n2 + bins + 1
        codes = flat[start : start + count]
        i, rest = codes % n0, codes // n0
        cells.append(np.stack([i - ng, rest % n1 - ng, rest // n1 - ng], axis=-1))
    return total, counts, cells


def stoichiometric():
    gas = ct.Solution(mechanism)
    gas.set_equivalence_ratio(1.0, "CH4", "O2")
    return gas.Y


def canterasdelay(p0, T0, Y, rise):
    gas = ct.Solution(mechanism)
    gas.TPY = T0, p0, Y
    reactor = ct.IdealGasReactor(gas)
    net = ct.ReactorNet([reactor])
    t = 0.0
    while reactor.T < T0 + rise:
        t = net.step()
    return t


def oursdelay(mb, dt, T0, rise, until):
    while mb.tme.get() < until:
        mb.step(dt)
        if temperature(mb) > T0 + rise:
            return mb.tme.get()
    return None


def randomstate(ns, rng):
    Y = rng.random(ns)
    return rng.uniform(1e5, 30e5), rng.uniform(600.0, 2800.0), Y / Y.sum()
