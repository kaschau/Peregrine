import numpy as np

import peregrine as pg
from peregrine.simulator.mixture import Ru

from tests.gases import primitives

# constants only, nothing from the library: two species of different weight, with the constants a viscous case's transport reads
gas = {
    "N2": {"MW": 28.0134, "cp0": 1040.0, "mu0": 1.78e-5, "kappa0": 0.0259},
    "O2": {"MW": 31.9988, "cp0": 918.0, "mu0": 2.05e-5, "kappa0": 0.0266},
}
# the wave's gas, which develop's reference errors were measured with
air = {"Air": {"MW": 28.97, "cp0": 1000.0}}
R = Ru / air["Air"]["MW"]

# every flux function under each reconstruction it takes, rusanov under each of muscl's other limiters, and the one blend
FLUXES = [
    dict(primary="kepaec", recon="constant"),
    dict(primary="kepaec4", recon="constant"),
    dict(primary="rusanov", recon="constant"),
    dict(primary="ausmup", recon="constant"),
    dict(primary="hllc", recon="constant"),
    dict(primary="kepaec", recon="muscl", limiter="koren"),
    dict(primary="rusanov", recon="muscl", limiter="koren"),
    dict(primary="ausmup", recon="muscl", limiter="koren"),
    dict(primary="hllc", recon="muscl", limiter="koren"),
    dict(primary="rusanov", recon="muscl", limiter="minmod"),
    dict(primary="rusanov", recon="muscl", limiter="mc"),
    dict(primary="rusanov", recon="muscl", limiter="superbee"),
    dict(primary="rusanov", recon="muscl", limiter="vanleer"),
]
BLENDED = dict(formulation="blended", primary="kepaec", secondary="rusanov", switch="jameson")


def fluxid(keys):
    return "-".join(str(v) for v in keys.values())


def advflux(config, **keys):
    for key, value in keys.items():
        config["adv-flux"][key] = value


def box(dims=(6, 6, 6), periodic=(False, False, False), physics="euler", mesher=pg.mesher.CubeMesher):
    config = pg.Config()
    config["simulation"]["simulator"] = physics
    config["simulation"]["dt"] = 1e-6
    config["time-integration"]["integrator"] = "rk1"
    config["mixture"]["species"] = gas
    config["mixture"]["eos"] = "cpg"
    config["mixture"]["trans"] = "constant" if physics == "navier-stokes" else None
    config["ics"]["Y"] = {"N2": 0.77}
    walled = not all(periodic)
    if walled:
        config["bcvals"]["wall"] = {"kind": "slp-adia-wall"}
    mesh = mesher(
        mbdims=[1, 1, 1],
        blkdims=list(dims),
        lengths=[1, 1, 1],
        periodic=list(periodic),
        defaultbc="wall" if walled else None,
    )
    return config, mesh


class JostledCube(pg.mesher.CubeMesher):
    def __init__(self, *args, jostle=0.03, seed=7, **kwargs):
        super().__init__(*args, **kwargs)
        self.jostle, self.seed = jostle, seed

    def shape_blk(self, blk, i, j, k):
        super().shape_blk(blk, i, j, k)
        rng = np.random.default_rng(self.seed)
        nodes = blk.nodes.get()
        ng = blk.ng
        # the block's outside planes stay, so a periodic halo lands on its partner's face and a wall's halo is its interior's reflection
        inside = np.s_[ng + 1 : -ng - 1, ng + 1 : -ng - 1, ng + 1 : -ng - 1]
        nodes[inside] += self.jostle * rng.uniform(-1.0, 1.0, nodes[inside].shape)
        blk.nodes.set(nodes)


def line(backend, nx, periodic=True, **keys):
    config = pg.Config()
    config["simulation"]["simulator"] = "euler"
    config["mixture"]["species"] = air
    config["mixture"]["eos"] = "cpg"
    config["time-integration"]["integrator"] = "tvd-rk3"
    config["bcvals"]["wall"] = {"kind": "slp-adia-wall"}
    advflux(config, **keys)
    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1],
        blkdims=[nx, 2, 2],
        lengths=[1, 0.1, 0.1],
        periodic=[periodic, False, False],
        defaultbc="wall",
    )
    return pg.get_solver(config, mesh, backend(config))


# the wave's pressure and speed, in SI: the energy carries the standard reference, under which a millikelvin state's thermal part would be lost
P0, U0 = 1e5, 400.0
RHO0 = P0 / U0**2


def carry_wave(backend, nx, u=1.0, **keys):
    mb = line(backend, nx, **keys)
    blk = mb.blocks[0]
    ng = blk.ng
    q = primitives(mb, blk)
    x = blk.xcc.get()[..., 0]
    q[..., 0], q[..., 1], q[..., 2:4] = P0, u * U0, 0.0
    q[..., 4] = P0 / (R * RHO0 * (2.0 + np.sin(2.0 * np.pi * x)))
    mb.set_prims([q])
    mass = totals(mb)[0]
    dt = 0.1 / (nx - 1) / U0
    n = 10 * (nx - 1)
    for _ in range(n):
        mb.step(dt)
    exact = 2.0 + np.sin(2.0 * np.pi * (x[ng:-ng, ng, ng] - u * U0 * n * dt))
    rho = blk.Q.get()[ng:-ng, ng, ng, 0] / RHO0
    prims = primitives(mb, blk)[ng:-ng, ng, ng]
    return np.abs(rho - exact), prims, totals(mb)[0] / mass - 1.0


def volumes(blk):
    return 1.0 / blk.detJinv.get()[blk.interior]


def totals(mb, name="Q"):
    blk = mb.blocks[0]
    q = getattr(blk, name).get()[blk.interior]
    return np.einsum("ijkl,ijk->l", q, volumes(blk))


def derivative(mb):
    blk = mb.blocks[0]
    stage = mb.integ.stageregs(mb.regs)[0]
    mb.rhs(Q=stage.Q, dQ=stage.dQ)
    return getattr(blk, stage.dQ).get()[blk.interior]


def fluxscale(blk):
    rows = []
    for a in "ijk":
        F = getattr(blk, f"{a}F").get()[blk.interior]
        rows.append(np.abs(F).max(axis=(0, 1, 2)))
    return np.max(rows, axis=0)


def wallfluxes(blk):
    total = 0.0
    for interface in blk.intfaces:
        F = getattr(blk, "ijk"[interface.axis] + "F").get()
        plane = interface.faceplane(F).sum(axis=(0, 1))
        total = total + (plane if interface.is_low else -plane)
    return total
