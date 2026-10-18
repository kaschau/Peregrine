import numpy as np
import pytest

import peregrine as pg
from peregrine.simulator.mixture import Tref

from tests.navstokes import support

# du_i/dx_j, dT/dx and the carried species' dY/dx; the last species' is what they leave
A = np.array([[3.0, -2.0, 1.0], [0.5, 4.0, -1.5], [-2.0, 1.0, 2.5]])
aT = np.array([20.0, -30.0, 10.0])
aY = np.array([[0.05, 0.1, -0.08], [-0.03, 0.06, 0.07]])
Y0 = np.array([0.5, 0.25])


def solver(backend, mesh, alpha=None):
    config = support.config(support.mixed)
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    if alpha is not None:
        config["diff-flux"]["values"] = {"alpha": alpha}
    return pg.get_solver(config, mesh, backend(config))


def diffflux(mb):
    blk = mb.blocks[0]
    mb.launch(mb.kernels["dqdxyz"]("Q"), "interior")
    for name in ("iF", "jF", "kF"):
        getattr(blk, name).set(np.zeros(getattr(blk, name).shape))
    mb.launch(mb.kernels["diffflux"]("Q"), "interior")


def fluxed_fields(mb, velocity):
    blk = mb.blocks[0]
    x = blk.xcc.get()
    u = 10.0 + x @ A.T if velocity else 0.0
    support.setprims(mb, blk, 101325.0, u, 300.0 + x @ aT, Y0 + x @ aY.T)
    diffflux(mb)


class Faces:
    def __init__(self, mb, axis):
        blk = mb.blocks[0]
        ng = blk.ng
        a = "ijk"[axis]
        F, S = getattr(blk, f"{a}F").get(), getattr(blk, f"{a}S").get()
        nf = F.shape[axis]
        # face f lies between cells f - 1 and f; both one in from the halo
        faces, left, right = ([slice(ng + 1, -ng - 1)] * 3 for _ in range(3))
        faces[axis] = slice(ng + 2, nf - ng - 2)
        left[axis] = slice(ng + 1, nf - ng - 3)
        right[axis] = slice(ng + 2, nf - ng - 2)
        self.F, self.S = F[tuple(faces)], S[tuple(faces)]
        L, R = tuple(left), tuple(right)
        Q, tr = blk.Q.get(), blk.trstate.get()
        T = support.statefield(mb, blk, "T")

        def mean(x):
            return 0.5 * (x[L] + x[R])

        # the halo's edges and corners are never written and hold no density
        with np.errstate(divide="ignore", invalid="ignore"):
            Y = Q[..., 5:] / Q[..., :1]
            u = Q[..., 1:4] / Q[..., :1]
        Yall = np.concatenate([Y, 1.0 - Y.sum(axis=-1, keepdims=True)], axis=-1)
        self.rho, self.T = mean(Q[..., 0]), mean(T)
        self.u, self.Y = mean(u), mean(Yall)
        self.mu, self.kappa, self.D = mean(tr[..., 0]), mean(tr[..., 1]), mean(tr[..., 2:])


def expected(faces, mb, work):
    S = faces.S
    mu = faces.mu[..., None, None]
    tau = -mu * (A + A.T) + 2.0 / 3.0 * mu * np.trace(A) * np.eye(3)
    momentum = np.einsum("...ij,...j->...i", tau, S)
    gY = S @ np.vstack([aY, -aY.sum(axis=0)]).T
    J = -faces.rho[..., None] * faces.D * gY
    Vc = (faces.D * gY).sum(axis=-1)
    J += faces.Y * (faces.rho * Vc)[..., None]
    species = [mb.sim.mixture.species[n] for n in mb.sim.mixture.spnames]
    cp = np.array([sp["cp0"] for sp in species])
    hf = np.array([sp["dHf298"] / sp["MW"] for sp in species])
    h = cp * (faces.T[..., None] - Tref) + hf
    energy = -faces.kappa * (S @ aT) + (J * h).sum(axis=-1)
    if work:
        energy += np.einsum("...i,...ij,...j->...", faces.u, tau, S)
    return momentum, energy, J[..., :-1]


def rel(got, want):
    return np.abs(got - want).max() / np.abs(want).max()


@pytest.fixture(scope="module")
def mb(backend):
    return solver(backend, support.box([9, 8, 7], [1.0, 1.0, 1.0], defaultbc="walls"))


@pytest.mark.parametrize("axis", range(3))
def test_momentum_rows_are_the_newtonian_stress(mb, axis):
    # measured 6e-14
    fluxed_fields(mb, velocity=True)
    faces = Faces(mb, axis)
    momentum, _, _ = expected(faces, mb, work=True)
    assert rel(faces.F[..., 1:4], momentum) < 1e-12


@pytest.mark.parametrize("axis", range(3))
def test_species_rows_are_ficks_law_with_the_correction(mb, axis):
    # three Lewis numbers, so the correction velocity is not zero; measured 3e-14
    fluxed_fields(mb, velocity=True)
    faces = Faces(mb, axis)
    _, _, species = expected(faces, mb, work=True)
    assert rel(faces.F[..., 5:], species) < 1e-12


# the formation enthalpies differ, so an enthalpy off the eos's reference shows
@pytest.mark.parametrize("axis", range(3))
def test_energy_row_at_rest_is_fourier_and_the_species_enthalpy(mb, axis):
    fluxed_fields(mb, velocity=False)
    faces = Faces(mb, axis)
    _, energy, _ = expected(faces, mb, work=False)
    assert rel(faces.F[..., 4], energy) < 1e-12


@pytest.mark.parametrize("axis", range(3))
def test_energy_row_carries_the_work_of_the_stress(mb, axis):
    fluxed_fields(mb, velocity=True)
    faces = Faces(mb, axis)
    _, energy, _ = expected(faces, mb, work=True)
    assert rel(faces.F[..., 4], energy) < 1e-12


# a checkerboard's central cell gradients are all zero, so the face gradient is alpha times the jump over the spacing
@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_alpha_damps_toward_the_two_point_gradient(backend, alpha):
    nx, L = 8, 1e-3
    mb = solver(backend, support.line(nx, L), alpha=alpha)
    blk = mb.blocks[0]
    ng = blk.ng
    T = 300.0 + 10.0 * (-1.0) ** np.arange(blk.Q.shape[0])
    support.setprims(mb, blk, 101325.0, 0.0, T[:, None, None], Y0)
    diffflux(mb)
    nf = blk.iF.shape[0]
    F = blk.iF.get()[ng : nf - ng, ng, ng, 4]
    area = np.abs(blk.iS.get()[ng : nf - ng, ng, ng, 0])
    kappa = blk.trstate.get()[ng, ng, ng, 1]
    dx = L / nx
    jump = T[ng : nf - ng] - T[ng - 1 : nf - ng - 1]
    want = -alpha * kappa * jump / dx * area
    scale = kappa * 20.0 / dx * area.max()
    assert np.abs(F - want).max() < 1e-12 * scale
