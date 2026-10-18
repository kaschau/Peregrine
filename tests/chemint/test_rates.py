import cantera as ct
import numpy as np
import pytest

from tests.mixture.rates import density, oneway_twin, production, troefiterr
from tests.chemint.cases import at_cell, cell, dQ, ownstate, randomstate, stoichiometric


@pytest.mark.parametrize(
    "mechanism, within",
    # measured 5e-6, 4e-4 and 4e-4 of the largest rate, the refit thermo's consequence on the reverse rates
    [("CH4_O2_FFCMY.yaml", 1e-4), ("C2H4_Air_Skeletal.yaml", 2e-3), ("GRI30.yaml", 2e-3)],
)
def test_the_rates_are_canteras(backend, mechanism, within, tmp_path):
    rng = np.random.default_rng(7)
    gas = ct.Solution(mechanism)
    # with every reaction one way no reverse rate carries the refit thermo, so every reaction body's share of each species' rate is checked
    twin = oneway_twin(gas, mechanism, tmp_path)
    oneway = str(tmp_path / mechanism)
    for trial in range(3):
        p, T, Y = randomstate(gas.n_species, rng)
        # a species that is the small net of large one-way terms holds to the terms' rounding: measured 5e-14 of the largest rate on the host, 7e-13 on an H100 with its own exp and log
        mb = cell(backend, oneway, "coupled", p, T, Y)
        omega = at_cell(mb, "omega")
        twin.TPY = T, p, Y
        theirs = twin.net_production_rates * twin.molecular_weights
        mixture = mb.sim.mixture
        exact = production(mixture, density(mixture, Y, T, p), Y, T, exact=True)
        assert np.all(np.abs(exact - theirs) <= 1e-10 * np.abs(theirs) + 1e-12 * np.abs(theirs).max())
        # the kernel's Troe centres are fitted: within the fit's budget of the largest rate besides
        fiterr = troefiterr(mixture)
        assert np.all(np.abs(omega - theirs) <= 1e-10 * np.abs(theirs) + (1e-12 + fiterr) * np.abs(theirs).max())
        # the twin on the cell's own state, the eos round trip having moved T by 6e-13; the product form's K exponent rounds at 1e-14 before its exp (measured 2.2e-14 over 800 states)
        assert np.abs(omega - production(mb.sim.mixture, *ownstate(mb))).max() < 3e-14 * np.abs(theirs).max()
        # the mechanism itself, reverse rates on the refit thermo
        mb = cell(backend, mechanism, "coupled", p, T, Y)
        omega = at_cell(mb, "omega")
        gas.TPY = T, p, Y
        theirs = gas.net_production_rates * gas.molecular_weights
        scale = np.abs(theirs).max()
        assert np.abs(omega - theirs).max() < (within + fiterr) * scale
        # a reaction near equilibrium cancels its forward and reverse rates, so a rounding in the affinity is 1e-11 of the net rate (measured up to 5e-11 with Apple's libm, 1.5e-10 for GRI30 with CUDA's)
        assert np.abs(omega - production(mb.sim.mixture, *ownstate(mb))).max() < 5e-10 * scale
        # mass is conserved: the rates sum to nothing
        assert abs(omega.sum()) < 1e-12 * scale
        # the source is the carried species' rates
        assert np.array_equal(dQ(mb)[5:], omega[:-1])


def test_single_follows_double(backend):
    # measured 1.1e-4 of the largest rate, an exponent rounded at float's ulp; the log-space form is what keeps single finite at all
    rng = np.random.default_rng(11)
    p, T, Y = randomstate(12, rng)
    mb = cell(backend, "CH4_O2_FFCMY.yaml", "coupled", p, T, Y)
    double = at_cell(mb, "omega")
    assert mb.kernels["rates"](Q=mb.regs[0]).text.count(", product form") == len(mb.sim.mixture.reactions)
    mb = cell(backend, "CH4_O2_FFCMY.yaml", "coupled", p, T, Y, precision="single")
    single = at_cell(mb, "omega")
    assert ", product form" not in mb.kernels["rates"](Q=mb.regs[0]).text
    assert np.isfinite(single).all()
    assert np.abs(single - double).max() < 1e-3 * np.abs(double).max()


def test_an_absent_species_rates_are_exactly_nothing(backend):
    # the product form: a one-way rate with an absent reactant is a product with a zero, where the log form's floor gave an e^-45-suppressed seed
    mb = cell(backend, "CH4_O2_FFCMY.yaml", "coupled", 10e5, 1200.0, stoichiometric())
    omega, names = at_cell(mb, "omega"), mb.sim.mixture.spnames
    assert np.isfinite(omega).all()
    assert omega[names.index("CH4")] < 0 and omega[names.index("O2")] < 0
    assert omega[names.index("CH3")] > 0 and omega[names.index("HO2")] > 0 and omega[names.index("H")] > 0
    # O comes of O2's dissociation, and the last species, CO2, carries one less the sum, a rounding above nothing, so its reactions run at 1e-27
    still = [names.index(n) for n in ("H2", "OH", "H2O", "CH2O")]
    assert np.array_equal(omega[still], np.zeros(len(still)))
    assert np.abs(omega[[names.index(n) for n in ("O", "CO", "CO2")]]).max() < 1e-20


def test_a_cold_cell_of_one_species_is_finite_and_still(backend):
    names = ct.Solution("CH4_O2_FFCMY.yaml").species_names
    Y = np.zeros(len(names))
    Y[names.index("O2")] = 1.0
    mb = cell(backend, "CH4_O2_FFCMY.yaml", "coupled", 1e5, 111.73, Y, trange=(100.0, 3500.0))
    omega = at_cell(mb, "omega")
    assert np.isfinite(omega).all() and np.array_equal(omega, np.zeros(len(names)))


def test_a_fractional_order_keeps_the_log_form(backend, tmp_path):
    # one reaction's order off its stoichiometry renders in the log form beside the product-form rest, the log state made for its species
    name = "CH4_O2_FFCMY.yaml"
    gas = ct.Solution(name)
    reactions = []
    changed = False
    for rxn in gas.reactions():
        rxn.reversible = False
        if not changed and "CH4" in rxn.reactants:
            rxn.orders = {"CH4": 1.5}
            changed = True
        reactions.append(rxn)
    twin = ct.Solution(thermo="ideal-gas", kinetics="gas", species=gas.species(), reactions=reactions)
    twin.write_yaml(str(tmp_path / name))
    rng = np.random.default_rng(3)
    p, T, Y = randomstate(gas.n_species, rng)
    mb = cell(backend, str(tmp_path / name), "coupled", p, T, Y)
    omega = at_cell(mb, "omega")
    assert mb.kernels["rates"](Q=mb.regs[0]).text.count(", product form") == len(reactions) - 1
    twin.TPY = T, p, Y
    scale = np.abs(twin.net_production_rates * twin.molecular_weights).max()
    assert np.abs(omega - production(mb.sim.mixture, *ownstate(mb))).max() < 3e-14 * scale
