from pathlib import Path

import cantera as ct
import numpy as np
import peregrine as pg

from tests.gases import primitives

mechanisms = ["CH4_O2_FFCMY.yaml", "GRI30.yaml", "C2H4_Air_Skeletal.yaml"]
# every fit of a mechanism is made over this; below it a kernel reads an extrapolation
trange = (300.0, 3500.0)


def cantera(ctfile):
    database = Path(pg.__file__).parent / "simulator" / "mixture" / "database"
    ct.add_directory(str(database / "mechanisms"))
    return ct.Solution(ctfile)


def mixture_config(species, precision="double", **choices):
    config = pg.Config()
    config["simulation"]["simulator"] = "navier-stokes"
    config["simulation"]["precision"] = precision
    mc = config["mixture"]
    mc["species"] = species
    for key, value in choices.items():
        mc[key] = value
    return config


def solver(backend, species, precision="double", **choices):
    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1], blkdims=[2, 2, 2], lengths=[1, 1, 1], periodic=[True, True, True]
    )
    config = mixture_config(species, precision, **choices)
    return pg.get_solver(config, mesh, backend(config))


def kinetic(backend, ctfile, **choices):
    choices = {"eos": "tpg", "trans": "kinetic-theory", "T-range": trange, **choices}
    return solver(backend, ctfile, **choices)


def draw(seed, nspecies, T=None):
    rng = np.random.default_rng(seed)
    p = rng.uniform(1e4, 1e6)
    T = rng.uniform(*trange) if T is None else T
    Y = rng.uniform(0.0, 1.0, nspecies)
    return p, T, Y / Y.sum()


def set_state(mb, p, T, Y):
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    q[..., 0] = p
    q[..., 4] = T
    q[..., 5:] = Y[:-1]
    mb.set_prims([q])
    ng = blk.ng
    # mu, kappa, then D per species
    return blk.trstate.get()[ng, ng, ng]


def check(want, got, tol, where):
    errors = {k: abs(got[k] / want[k] - 1.0) for k in want}
    worst = max(errors, key=errors.get)
    rows = "\n".join(
        f"{k:>10s}: {want[k]:16.8e} {got[k]:16.8e} {errors[k]:9.2e}" for k in want
    )
    print(f"{where}\n{'':>10s}  {'reference':>16s} {'peregrine':>16s} {'rel':>9s}\n{rows}")
    assert errors[worst] < tol, f"{where}: {worst} off by {errors[worst]:.2e}"


def molefrac(Y, MW):
    X = Y / MW
    return X / X.sum()


def wilke(mu, X, MW):
    ratio = MW[None, :] / MW[:, None]
    phi = (1.0 + np.sqrt(mu[:, None] / mu[None, :]) * ratio**0.25) ** 2
    phi /= np.sqrt(8.0 * (1.0 + 1.0 / ratio))
    return np.sum(X * mu / (phi @ X))


def herning(mu, X, MW):
    return np.sum(X * mu * np.sqrt(MW)) / np.sum(X * np.sqrt(MW))


rules = {"wilke": wilke, "herning": herning}


def kappamix(kappa, X):
    return 0.5 * (np.sum(X * kappa) + 1.0 / np.sum(X / kappa))


# the fits are in ln T, of sqrt(mu / sqrt(T)) and of kappa / sqrt(T)
def species_fits(constants, T):
    lnT = np.log(T)
    mu = np.array([np.polyval(np.flip(row), lnT) for row in constants["mupoly"]])
    kappa = np.array([np.polyval(np.flip(row), lnT) for row in constants["kappapoly"]])
    return mu**2 * np.sqrt(T), kappa * np.sqrt(T)


# the fits are in ln T of T^1.5 / D at unit pressure, over the upper triangle
def pair_fits(constants, T):
    pairs = np.array([np.polyval(np.flip(row), np.log(T)) for row in constants["dijinv"]])
    ns = constants["MW"].shape[0]
    D = np.zeros((ns, ns))
    i, j = np.triu_indices(ns)
    D[i, j] = D[j, i] = T**1.5 / pairs
    return D
