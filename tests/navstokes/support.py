import numpy as np

import peregrine as pg

from tests.gases import primitives

# two species alike in weight and heat capacity, so a layer of one in the other diffuses at uniform density and temperature; air's transport
alike = {
    "A": {"MW": 28.0, "cp0": 1000.0, "mu0": 1.86e-5, "kappa0": 0.0263},
    "B": {"MW": 28.0, "cp0": 1000.0, "mu0": 1.86e-5, "kappa0": 0.0263},
}

# three species differing in everything the diffusive flux reads: weight, heat capacity, formation enthalpy, Lewis number, viscosity, conductivity
mixed = {
    "A": {"MW": 28.0, "cp0": 1000.0, "dHf298": 0.0, "lewis": 1.0, "mu0": 1.8e-5, "kappa0": 0.026},
    "B": {"MW": 32.0, "cp0": 900.0, "dHf298": 1.0e7, "lewis": 1.4, "mu0": 2.0e-5, "kappa0": 0.025},
    "C": {"MW": 16.0, "cp0": 2200.0, "dHf298": -7.5e7, "lewis": 0.7, "mu0": 1.1e-5, "kappa0": 0.034},
}


def config(species, eos="cpg"):
    config = pg.Config()
    config["simulation"]["simulator"] = "navier-stokes"
    mixture = config["mixture"]
    mixture["species"] = species
    mixture["eos"] = eos
    mixture["trans"] = "constant"
    mixture["diffusion"] = "lewis"
    config["adv-flux"]["primary"] = "kepaec"
    config["adv-flux"]["recon"] = "constant"
    config["time-integration"]["integrator"] = "tvd-rk3"
    return config


def box(dims, lengths, periodic=(False, False, False), **names):
    return pg.mesher.CubeMesher(
        mbdims=[1, 1, 1], blkdims=list(dims), lengths=list(lengths), periodic=list(periodic), **names
    )


def line(nx, L, periodic=True):
    dx = L / nx
    across = [True, True] if periodic else [False, False]
    return box([nx + 1, 2, 2], [L, dx, dx], periodic=[periodic, *across])


class SmoothCube(pg.mesher.CubeMesher):
    def __init__(self, amp=0.05, jostle=0.0, seed=11, **kwargs):
        super().__init__(**kwargs)
        self.amp, self.jostle, self.seed = amp, jostle, seed

    def shape_blk(self, blk, i, j, k):
        super().shape_blk(blk, i, j, k)
        nodes = blk.nodes.get()
        inner = nodes[blk.interior]
        xi, eta, zeta = (inner[..., c].copy() for c in range(3))
        two_pi = 2.0 * np.pi
        inner[..., 0] = xi + self.amp * np.sin(two_pi * xi) * np.cos(two_pi * eta)
        inner[..., 1] = eta + self.amp * np.sin(two_pi * eta) * np.cos(two_pi * zeta)
        inner[..., 2] = zeta + self.amp * np.sin(two_pi * zeta) * np.cos(two_pi * xi)
        if self.jostle:
            rng = np.random.default_rng(self.seed)
            h = 1.0 / (self.blkdims[0] - 1)
            core = inner[1:-1, 1:-1, 1:-1]
            core += rng.uniform(-1.0, 1.0, core.shape) * self.jostle * h
        nodes[blk.interior] = inner
        blk.nodes.set(nodes)


def statefield(mb, blk, name):
    return blk.state.get()[..., mb.sim.statefields.index(name)]


def volumes(blk):
    return 1.0 / blk.detJinv.get()[blk.interior]


def totals(mb):
    return sum((blk.Q.get()[blk.interior] * volumes(blk)[..., None]).sum(axis=(0, 1, 2)) for blk in mb.blocks)


def setprims(mb, blk, p, u, T, Y):
    q = primitives(mb, blk)
    q[..., 0], q[..., 1:4], q[..., 4] = p, u, T
    if Y is not None:
        q[..., 5:] = Y
    mb.set_prims([q])


def wallfaces(blk, axis):
    ng = blk.ng
    nf = getattr(blk, f"{'ijk'[axis]}F").shape[axis]
    return [(ng, ng, -1.0), (nf - ng - 1, nf - ng - 2, 1.0)]


def interior_plane(array, axis, index, ng):
    plane = np.take(array, index, axis=axis)
    return plane[ng:-ng, ng:-ng]
