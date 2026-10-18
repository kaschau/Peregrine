import numpy as np
import pytest

from peregrine.simulator.mixture import Ru, database
from peregrine.simulator.mixture.polyfit import PolyFitMixin

from tests.dense import support


# Poling's Chung as one nonpolar fluid: pairs 9-5.24 to 9-5.44, dense viscosity 9-6.16 to 9-6.18 and conductivity 10-5.5, each over its dilute limit; cv in J/mol/K, rho in kg/m3
def enhancement(species, X, T, rho, cv):
    chung = database("chung")
    E, B = np.array(chung["viscosity"]), np.array(chung["conductivity"])
    MW = np.array([sp["MW"] for sp in species])
    Vc = np.array([sp["Vcrit"] for sp in species]) * MW * 1e3
    sigma = 0.809 * Vc ** (1.0 / 3.0)
    epsk = np.array([sp["Tcrit"] for sp in species]) / 1.2593
    w = np.array([sp["acentric"] for sp in species])
    xx = np.outer(X, X)
    sigmaij = np.sqrt(np.outer(sigma, sigma))
    epsij = np.sqrt(np.outer(epsk, epsk))
    Mij = 2.0 * np.outer(MW, MW) / np.add.outer(MW, MW)
    sigma3 = np.sum(xx * sigmaij**3)
    epskm = np.sum(xx * epsij * sigmaij**3) / sigma3
    Mm = (np.sum(xx * epsij * sigmaij**2 * np.sqrt(Mij)) / (epskm * sigma3 ** (2.0 / 3.0))) ** 2
    wm = np.sum(xx * 0.5 * np.add.outer(w, w) * sigmaij**3) / sigma3
    Tc, Vcm = 1.2593 * epskm, sigma3 / 0.809**3
    Fc = 1.0 - 0.2756 * wm
    E = E[:, 0] + E[:, 1] * wm
    B = B[:, 0] + B[:, 1] * wm
    Tr, Tstar = T / Tc, T / epskm
    Omega = (
        1.16145 * Tstar**-0.14874
        + 0.52487 * np.exp(-0.7732 * Tstar)
        + 2.16178 * np.exp(-2.43787 * Tstar)
    )
    y = rho / Mm * 1e-3 * Vcm / 6.0
    G1 = (1.0 - 0.5 * y) / (1.0 - y) ** 3

    def G(c):
        return (c[0] * (1.0 - np.exp(-c[3] * y)) / y + c[1] * G1 * np.exp(c[4] * y) + c[2] * G1) / (
            c[0] * c[3] + c[1] + c[2]
        )

    G2 = G(E)
    etastarstar = E[6] * y**2 * G2 * np.exp(E[7] + E[8] / Tstar + E[9] / Tstar**2)
    muratio = (1.0 / G2 + E[5] * y) + etastarstar * Omega / (np.sqrt(Tstar) * Fc)
    Mprime = Mm * 1e-3
    eta0 = 4.0785e-6 * np.sqrt(Mm * T) / (Vcm ** (2.0 / 3.0) * Omega) * Fc
    alpha = cv / (Ru / 1000.0) - 1.5
    beta = 0.7862 - 0.7109 * wm + 1.3168 * wm**2
    Z = 2.0 + 10.5 * Tr**2
    Psi = 1.0 + alpha * (
        (0.215 + 0.28288 * alpha - 1.061 * beta + 0.26665 * Z)
        / (0.6366 + beta * Z + 1.061 * alpha * beta)
    )
    lambda0 = 31.2 * eta0 * Psi / Mprime
    H2 = G(B)
    kapparatio = (1.0 / H2 + B[5] * y) + 3.586e-3 * np.sqrt(Tc / Mprime) / Vcm ** (
        2.0 / 3.0
    ) * B[6] * y**2 * H2 * np.sqrt(Tr) / lambda0
    return muratio, kapparatio


def idealcv(species, X, T):
    lnT = np.log(T)
    cvR = sum(x * (PolyFitMixin.evaluate(sp["cppoly"], lnT) - 1.0) for x, sp in zip(X, species))
    return cvR * Ru / 1000.0


# few solvers: each loads its kernels into this process, and macOS caps the thread-local keys they take
@pytest.fixture(scope="module")
def mb(backend):
    return support.solver(backend, ["O2", "N2", "H2"], trans="chung", trange=(100.0, 1000.0))


@pytest.fixture(scope="module")
def kinetic(backend):
    return support.solver(backend, ["O2", "N2", "H2"], trange=(100.0, 1000.0))


@pytest.fixture(scope="module")
def wet(backend):
    return support.solver(backend, ["O2", "H2O", "CH4", "N2"], trans="chung", trange=(100.0, 1000.0))


def test_at_low_density_it_is_kinetic_theory(mb, kinetic):
    # a ten-thousandth of the enhancement is what 1 atm leaves
    Y = np.array([1.0, 0.0, 0.0])
    _, (mu, kappa, *_) = support.transport(mb, support.atm, 300.0, Y)
    _, (muk, kappak, *_) = support.transport(kinetic, support.atm, 300.0, Y)
    assert mu == pytest.approx(muk, rel=1e-3) and kappa == pytest.approx(kappak, rel=1e-3)
    # and kinetic theory is oxygen's: NIST (Lemmon and Jacobsen 2004) at 300 K, 1 atm
    assert muk == pytest.approx(20.65e-6, rel=0.03)
    assert kappak == pytest.approx(0.02658, rel=0.03)


# NIST at 300 K, 1 atm: the viscosity of Muzny, Huber and Kazakov (2013), the conductivity of Assael et al. (2011)
def test_hydrogen_is_nists(kinetic):
    _, (mu, kappa, *_) = support.transport(kinetic, support.atm, 300.0, np.array([0.0, 0.0, 1.0]))
    assert mu == pytest.approx(8.94e-6, rel=0.03)
    assert kappa == pytest.approx(0.187, rel=0.05)


# the ideal cv comes off the case's own fits, which the conductivity's ratio feels through Psi; measured 7e-13
@pytest.mark.parametrize("X", [(1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.2, 0.8, 0.0)])
def test_at_high_density_it_is_kinetic_theory_times_chungs_ratio(mb, kinetic, X):
    species = support.species(mb)
    X = np.array(X)
    Y = support.massfrac(species, X)
    for p, T in ((60e6, 300.0), (60e6, 784.2), (30e6, 200.0)):
        rho, (mu, kappa, *_) = support.transport(mb, p, T, Y)
        _, (muk, kappak, *_) = support.transport(kinetic, support.atm, T, Y)
        muratio, kapparatio = enhancement(species, X, T, rho, idealcv(species, X, T))
        assert mu == pytest.approx(muk * muratio, rel=1e-6), (X, p, T)
        assert kappa == pytest.approx(kappak * kapparatio, rel=1e-6), (X, p, T)


# the one-fluid dipole rule would put 3.8 % water 20 % above pure oxygen; dilute kinetic theory puts it a few percent below
def test_a_little_water_lowers_oxygens_viscosity(wet):
    p, T = 59.761e6, 772.8
    _, (dry, *_) = support.transport(wet, p, T, np.array([1.0, 0.0, 0.0, 0.0]))
    _, (mixed, *_) = support.transport(wet, p, T, np.array([0.962, 0.038, 0.0, 0.0]))
    assert 0.93 < mixed / dry < 1.0


# liquid-like oxygen, where methane and water at the mixture's density would have y above one
def test_a_species_not_there_changes_nothing(mb, wet):
    p, T = 59.761e6, 111.73
    rho, (mu, kappa, *_) = support.transport(mb, p, T, np.array([1.0, 0.0, 0.0]))
    rho2, (mu2, kappa2, *_) = support.transport(wet, p, T, np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.isfinite(mu) and np.isfinite(kappa)
    # to the last bits of the temperature the state comes back with
    assert (rho2, mu2, kappa2) == pytest.approx((rho, mu, kappa), rel=1e-12)
