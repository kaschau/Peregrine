import pytest

from tests.transport import cases
from tests.transport.test_kinetic import states


@pytest.mark.parametrize("seed, T", states)
@pytest.mark.parametrize("ctfile", cases.mechanisms)
def test_unitylewis(backend, ctfile, seed, T):
    gas = cases.cantera(ctfile)
    p, T, Y = cases.draw(seed, gas.n_species, T)
    gas.TPY = T, p, Y
    mb = cases.kinetic(backend, ctfile, diffusion="lewis")
    assert mb.sim.mixture.trans.name == "kinetic-theory"
    assert mb.sim.mixture.diffusion.name == "lewis"
    got = cases.set_state(mb, p, T, Y)
    want = {"mu": gas.viscosity, "kappa": gas.thermal_conductivity}
    alpha = gas.thermal_conductivity / (gas.density * gas.cp_mass)
    for name in gas.species_names:
        want[f"D_{name}"] = alpha
    got = dict(zip(want, got))
    # the worst in-range row over 500 seeds is 0.6 %; a draw below the range reaches 3 %
    cases.check(want, got, 1e-2, f"{ctfile} seed {seed}: p {p:.4e} T {T:.2f}")
