import numpy as np

import peregrine as pg

from tests.gases import primitives

atm = 101325.0


def solver(backend, species, trans="kinetic-theory", diffusion="lewis", trange=(100.0, 1500.0), mesh=None, bcvals=None):
    config = pg.Config()
    config["mixture"]["species"] = species
    config["mixture"]["eos"] = "real-gas"
    config["mixture"]["trans"] = trans
    config["mixture"]["diffusion"] = diffusion
    config["mixture"]["T-range"] = trange
    config["simulation"]["simulator"] = "navier-stokes"
    config["bcvals"].update(bcvals or {})
    if mesh is None:
        mesh = pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[2, 2, 2], lengths=[1, 1, 1], periodic=[True, True, True])
    return pg.get_solver(config, mesh, backend(config))


# the trstate row is mu, kappa, then D of every species
def transport(mb, p, T, Y):
    blk = mb.blocks[0]
    ng = blk.ng
    q = primitives(mb, blk)
    q[..., 0], q[..., 1:4], q[..., 4], q[..., 5:] = p, 0.0, T, np.asarray(Y)[:-1]
    mb.set_prims([q])
    return blk.Q.get()[ng, ng, ng, 0], blk.trstate.get()[ng, ng, ng]


def species(mb):
    return [mb.sim.mixture.species[n] for n in mb.sim.mixture.spnames]


def massfrac(species, X):
    MW = np.array([sp["MW"] for sp in species])
    return X * MW / np.sum(X * MW)
