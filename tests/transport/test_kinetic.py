import pytest

from tests.transport import cases

# the fitted range's ends are its weakest rows, so two states sit on them
states = [(1, None), (2, None), (3, cases.trange[0]), (4, cases.trange[1])]


@pytest.mark.parametrize("seed, T", states)
@pytest.mark.parametrize("ctfile", cases.mechanisms)
def test_kinetic(backend, ctfile, seed, T):
    gas = cases.cantera(ctfile)
    p, T, Y = cases.draw(seed, gas.n_species, T)
    gas.TPY = T, p, Y
    mb = cases.kinetic(backend, ctfile, diffusion="binary")
    assert mb.sim.mixture.trans.name == "kinetic-theory"
    assert mb.sim.mixture.diffusion.name == "binary"
    got = cases.set_state(mb, p, T, Y)
    want = {"mu": gas.viscosity, "kappa": gas.thermal_conductivity}
    for name, D in zip(gas.species_names, gas.mix_diff_coeffs_mass):
        want[f"D_{name}"] = D
    got = dict(zip(want, got))
    # the worst in-range row over 500 seeds is 0.6 %; a draw below the range reaches 4 %
    cases.check(want, got, 1e-2, f"{ctfile} seed {seed}: p {p:.4e} T {T:.2f}")
