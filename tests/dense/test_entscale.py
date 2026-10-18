import json

import CoolProp.CoolProp as CP
import numpy as np
import pytest

from peregrine.simulator.mixture import Ru
from peregrine.simulator.mixture.eos import RealGasModel

from tests.dense import support

species = ["O2", "N2", "CH4", "H2", "H2O", "CO2", "AR"]
coolprop = {
    "O2": "Oxygen",
    "N2": "Nitrogen",
    "CH4": "Methane",
    "H2": "Hydrogen",
    "H2O": "Water",
    "CO2": "CarbonDioxide",
    "AR": "Argon",
}
kB, NA = 1.380649e-23, 6.02214076e26


# s+ = -s_res / R of the cubic with kij = 0, at the cubic's own volume with the species' translation put back
def splus(sp, X, T, rho):
    c = RealGasModel.cubic
    MW = np.array([s["MW"] for s in sp])
    Tc = np.array([s["Tcrit"] for s in sp])
    pc = np.array([s["pcrit"] for s in sp])
    w = np.array([s["acentric"] for s in sp])
    MWmix = np.sum(X * MW)
    m = c["m"][0] + c["m"][1] * w + c["m"][2] * w**2
    sqrtac = np.sqrt(c["omega_a"] / pc) * Ru * Tc
    sqrta = sqrtac * np.abs(1.0 + m * (1.0 - np.sqrt(T / Tc)))
    S1, S2 = np.sum(X * sqrta), np.sum(X * sqrtac * m / np.sqrt(Tc))
    dam = -S1 * S2 / np.sqrt(T)
    bm = np.sum(X * c["omega_b"] * Ru * Tc / pc)
    V = MWmix / rho + np.sum(X * np.array([s.get("vshift", 0.0) for s in sp]))
    return -(np.log(1.0 - bm / V) + dam / (bm * Ru) * np.log(1.0 + bm / V))


def residual(sp, X, T, rho):
    MW = np.array([s["MW"] for s in sp])
    Y = support.massfrac(sp, X)
    MWmix = np.sum(X * MW)
    s = splus(sp, X, T, rho)
    assert s > 0.0
    nmu = np.array([sp_["resmu"] for sp_ in sp]).T @ X
    nkappa = np.array([sp_["reskappa"] for sp_ in sp]).T @ X
    rhoN = rho * NA / MWmix
    etaplus = np.exp(np.sum(nmu * s ** np.array([1.8, 2.4, 2.8]))) - 1.0
    mu = etaplus * rhoN ** (2.0 / 3.0) * np.sqrt(MWmix / NA * kB * T) / s ** (2.0 / 3.0)
    lambdaplus = np.sum(nkappa * s ** np.array([1.0, 1.5, 2.0, 2.5]))
    invsqrtmass = np.sum(Y / np.sqrt(MW / NA))
    kappa = lambdaplus * rhoN ** (2.0 / 3.0) * kB * np.sqrt(kB * T) * invsqrtmass / s ** (2.0 / 3.0)
    return mu, kappa


# few solvers: each loads its kernels into this process, and macOS caps the thread-local keys they take
@pytest.fixture(scope="module")
def mb(backend):
    return support.solver(backend, species, trans="entropy-scaling")


@pytest.fixture(scope="module")
def kinetic(backend):
    return support.solver(backend, species)


def pure(name):
    X = np.zeros(len(species))
    X[species.index(name)] = 1.0
    return X


@pytest.mark.parametrize("name", ["O2", "N2", "CH4", "CO2", "AR"])
def test_at_low_density_it_is_kinetic_theory(mb, kinetic, name):
    # what a residual in s+^1.8 and s+ over s+^(2/3) leaves at 1 atm
    Y = pure(name)
    _, (mu, kappa, *_) = support.transport(mb, support.atm, 300.0, Y)
    _, (muk, kappak, *_) = support.transport(kinetic, support.atm, 300.0, Y)
    assert mu == pytest.approx(muk, rel=1e-3)
    assert kappa == pytest.approx(kappak, rel=1e-2)


mixtures = {
    "O2": (pure("O2"), ((60e6, 300.0), (60e6, 800.0), (30e6, 200.0))),
    "N2": (pure("N2"), ((60e6, 300.0), (60e6, 800.0), (30e6, 200.0))),
    "CH4": (pure("CH4"), ((60e6, 300.0), (60e6, 800.0), (30e6, 200.0))),
    "H2": (pure("H2"), ((60e6, 300.0), (60e6, 800.0), (30e6, 200.0))),
    "CO2": (pure("CO2"), ((60e6, 300.0), (60e6, 800.0), (30e6, 400.0))),
    "AR": (pure("AR"), ((60e6, 300.0), (60e6, 800.0), (30e6, 200.0))),
    "H2O": (pure("H2O"), ((60e6, 700.0), (60e6, 1200.0), (30e6, 900.0))),
    "O2/N2": (0.5 * pure("O2") + 0.5 * pure("N2"), ((60e6, 300.0), (60e6, 800.0))),
    "CO2/N2": (0.5 * pure("CO2") + 0.5 * pure("N2"), ((60e6, 300.0), (60e6, 800.0))),
    "O2/H2": (0.9 * pure("O2") + 0.1 * pure("H2"), ((60e6, 300.0), (60e6, 800.0))),
    "H2O/CO2": (0.5 * pure("H2O") + 0.5 * pure("CO2"), ((60e6, 700.0), (60e6, 1200.0))),
}


@pytest.mark.parametrize("name", list(mixtures))
def test_at_high_density_it_is_kinetic_theory_plus_the_residual(mb, kinetic, name):
    X, states = mixtures[name]
    sp = support.species(mb)
    Y = support.massfrac(sp, X)
    for p, T in states:
        rho, (mu, kappa, *_) = support.transport(mb, p, T, Y)
        _, (muk, kappak, *_) = support.transport(kinetic, p, T, Y)
        mures, kappares = residual(sp, X, T, rho)
        assert mu == pytest.approx(muk + mures, rel=1e-6), (name, p, T)
        assert kappa == pytest.approx(kappak + kappares, rel=1e-6), (name, p, T)


def srk(sp, name):
    fluid = f"PG_{name}"
    try:
        return CP.AbstractState("SRK", fluid)
    except ValueError:
        pass
    MW = sp["MW"]
    CP.add_fluids_as_JSON(
        "SRK",
        json.dumps(
            [
                dict(
                    name=fluid,
                    CAS=fluid,
                    aliases=[],
                    Tc=sp["Tcrit"],
                    Tc_units="K",
                    pc=sp["pcrit"],
                    pc_units="Pa",
                    acentric=sp["acentric"],
                    molemass=MW / 1e3,
                    molemass_units="kg/mol",
                    rhomolarc=1.0 / (sp["Vcrit"] * MW / 1e3),
                    rhomolarc_units="mol/m^3",
                )
            ]
        ),
    )
    return CP.AbstractState("SRK", fluid)


# CoolProp's smolar_residual is the Helmholtz residual at (T, v), not an entropy at (T, p) less an ideal part, which is R ln Z off; the two codes' omega_a differ by 2e-5, which leaves 6e-7 here
@pytest.mark.parametrize("name", ["O2", "N2", "CH4", "H2", "H2O", "CO2", "AR"])
def test_the_residual_entropy_is_the_cubics(mb, name):
    sp = support.species(mb)
    X = pure(name)
    s = sp[species.index(name)]
    AS = srk(s, name)
    for p, T in mixtures[name][1][:2]:
        rho, _ = support.transport(mb, p, T, support.massfrac(sp, X))
        v = s["MW"] / rho + s.get("vshift", 0.0)
        AS.update(CP.DmolarT_INPUTS, 1e3 / v, T)
        assert splus(sp, X, T, rho) == pytest.approx(-AS.smolar_residual() / 8.314462618, rel=2e-6), (name, p, T)


# states away from the critical point within the fits' range, with tolerances at the model's own reach: 2-4 % for these fluids, 7-13 % for water and hydrogen, and water's gas-like conductivity 20-25 % above IAPWS from Mason-Monchick's dilute part
references = [
    ("O2", 125.0, 60e6, 0.08, 0.08),
    ("O2", 300.0, 60e6, 0.08, 0.08),
    ("O2", 230.0, 20e6, 0.08, 0.08),
    ("N2", 300.0, 60e6, 0.08, 0.08),
    ("N2", 190.0, 20e6, 0.08, 0.08),
    ("CH4", 300.0, 60e6, 0.08, 0.08),
    ("CH4", 160.0, 60e6, 0.08, 0.08),
    ("CO2", 600.0, 60e6, 0.08, 0.08),
    ("CO2", 450.0, 20e6, 0.08, 0.08),
    ("AR", 300.0, 60e6, 0.08, 0.08),
    ("AR", 225.0, 20e6, 0.08, 0.08),
    ("H2", 120.0, 60e6, 0.15, 0.15),
    ("H2", 300.0, 60e6, 0.15, 0.15),
    ("H2O", 700.0, 60e6, 0.15, 0.15),
    ("H2O", 1000.0, 20e6, 0.15, 0.30),
]


@pytest.mark.parametrize("name, T, p, tolmu, tolkappa", references)
def test_pure_fluids_follow_cool_prop(mb, name, T, p, tolmu, tolkappa):
    _, (mu, kappa, *_) = support.transport(mb, p, T, pure(name))
    fluid = coolprop[name]
    muref = CP.PropsSI("V", "T", T, "P", p, fluid)
    kapparef = CP.PropsSI("L", "T", T, "P", p, fluid)
    assert mu == pytest.approx(muref, rel=tolmu), (name, T, p, mu, muref)
    assert kappa == pytest.approx(kapparef, rel=tolkappa), (name, T, p, kappa, kapparef)
