import numpy as np
import pytest

from tests.transport import cases

pytestmark = pytest.mark.parametrize("ctfile", ["CH4_O2_FFCMY.yaml", "GRI30.yaml"])


def mixed(backend, ctfile, rule, seed):
    gas = cases.cantera(ctfile)
    p, T, Y = cases.draw(seed, gas.n_species)
    gas.TPY = T, p, Y
    choices = {"diffusion": "lewis"}
    if rule:
        choices["mixing-rule"] = rule
    mb = cases.kinetic(backend, ctfile, **choices)
    mu = cases.set_state(mb, p, T, Y)[0]
    constants = mb.sim.mixture.constants()
    musp, _ = cases.species_fits(constants, T)
    X = cases.molefrac(Y, constants["MW"])
    name = mb.sim.mixture.mixrule.name
    return mu, cases.rules[name](musp, X, constants["MW"]), gas, name


def test_wilke(backend, ctfile):
    # the default rule; the pair constant and MW^(-1/4) are baked in the kernel
    mu, numpy, gas, name = mixed(backend, ctfile, None, 5)
    assert name == "wilke"
    # the kernel and numpy agree to 1.4e-12 today; the baked constants transposed move mu by percents
    print(
        f"mu: wilke {mu:.6e} | numpy wilke {numpy:.6e} ({abs(mu / numpy - 1):.1e}) "
        f"| cantera wilke {gas.viscosity:.6e} ({abs(mu / gas.viscosity - 1) * 100:.2f} %)"
    )
    assert mu == pytest.approx(numpy, rel=1e-10)


def test_herning(backend, ctfile):
    mu, numpy, gas, name = mixed(backend, ctfile, "herning", 6)
    assert name == "herning"
    assert mu == pytest.approx(numpy, rel=1e-10)
    X, sqrtMW = gas.X, np.sqrt(gas.molecular_weights)
    herning = np.sum(X * gas.species_viscosities * sqrtMW) / np.sum(X * sqrtMW)
    print(
        f"mu: herning {mu:.6e} | herning over cantera {herning:.6e} "
        f"({abs(mu - herning) / herning * 100:.2e} %) | cantera wilke {gas.viscosity:.6e} "
        f"({abs(mu - gas.viscosity) / gas.viscosity * 100:.2f} %)"
    )
    # the species fits are within refit-tol of kinetic theory, and Cantera's of its own
    assert abs(mu - herning) / herning < 1e-2
