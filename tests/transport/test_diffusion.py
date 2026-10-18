import numpy as np
import pytest

from tests.transport import cases
from tests.transport.test_constant import two

ctfile = "CH4_O2_FFCMY.yaml"


@pytest.mark.parametrize("lone", [0, 5, 11])
def test_pure_cell_binary(backend, lone):
    # the last species is what the carried ones leave, its own path in the kernel
    gas = cases.cantera(ctfile)
    ns = gas.n_species
    assert lone < ns
    mb = cases.kinetic(backend, ctfile, diffusion="binary")
    p, T, _ = cases.draw(10, ns)
    Y = np.zeros(ns)
    Y[lone] = 1.0
    D = cases.set_state(mb, p, T, Y)[2:]
    pairs = cases.pair_fits(mb.sim.mixture.constants(), T)
    # the lone species, with nothing to take the trace limit in, takes its self-diffusion coefficient
    assert D[lone] == pytest.approx(pairs[lone, lone] / p, rel=1e-12)
    others = np.arange(ns) != lone
    # the others' mixture-averaged formula collapses to the pair with the lone species
    assert D[others] == pytest.approx(pairs[others, lone] / p, rel=1e-12)
    gas.TPY = T, p, Y
    # the pair fits are within 0.7 % of Cantera's
    assert D[others] == pytest.approx(gas.binary_diff_coeffs[others, lone], rel=1e-2)


# a trace inside each precision's normal range and far below the kernel's tolerance of 64 epsilon of the whole, and one above it that counts; single's bound is the reciprocal fit's Horner in an uncentred ln T at float32
@pytest.mark.parametrize("precision, trace, counts, rel", [("double", 1e-44, 1e-9, 1e-12), ("single", 1e-30, 1e-5, 1e-4)])
def test_traces_of_either_sign_read_as_the_pure_cell(backend, precision, trace, counts, rel):
    # traces of both signs beside a pure species: the negative ones count as absent and so do positive ones under the tolerance, leaving the pure cell; a trace over the tolerance is what the dominant species diffuses in
    gas = cases.cantera(ctfile)
    ns = gas.n_species
    names = gas.species_names
    mb = cases.kinetic(backend, ctfile, diffusion="binary", precision=precision)
    p, T, _ = cases.draw(10, ns)
    pairs = cases.pair_fits(mb.sim.mixture.constants(), T)
    # the dominant species is the implied last one: a carried one comes back off the solution as rho Y / rho, leaving the implied species a trace at the ulp, far above these
    lone, up = ns - 1, names.index("CH3")
    Y = np.zeros(ns)
    Y[lone] = 1.0
    Y[names.index("O2")], Y[up], Y[names.index("CO")] = -3 * trace, 5 * trace, -0.8 * trace
    D = cases.set_state(mb, p, T, Y)[2:]
    assert np.all(np.isfinite(D))
    others = np.arange(ns) != lone
    assert D[others] == pytest.approx(pairs[others, lone] / p, rel=rel)
    assert D[lone] == pytest.approx(pairs[lone, lone] / p, rel=rel)
    # negative traces alone: the pure cell
    Y[up] = -5 * trace
    D = cases.set_state(mb, p, T, Y)[2:]
    assert D[lone] == pytest.approx(pairs[lone, lone] / p, rel=rel)
    assert D[others] == pytest.approx(pairs[others, lone] / p, rel=rel)
    # a trace that counts sets the direction: the dominant species' pair with it
    Y[up] = counts
    D = cases.set_state(mb, p, T, Y)[2:]
    assert D[lone] == pytest.approx(pairs[lone, up] / p, rel=rel)


def test_a_bath_sums_over_its_species(backend):
    # the partners a coefficient sums over are the bath's, the first sum scaled from the bath's fraction to the whole, as if what the bath leaves out diffused as the bath does; against the rule written out on the case's own pair fits
    gas = cases.cantera(ctfile)
    ns, names = gas.n_species, gas.species_names
    bath = ["CH4", "O2", "H2O", "CO2"]
    mb = cases.kinetic(backend, ctfile, diffusion="binary", bath=bath)
    c = mb.sim.mixture.constants()
    assert list(c["bath"]) == [names.index(n) for n in bath]
    p, T, Y = cases.draw(10, ns)
    D = cases.set_state(mb, p, T, Y)[2:]
    pairs = cases.pair_fits(c, T)
    MW = c["MW"]
    X = Y / MW
    X /= X.sum()
    for n in range(ns):
        partners = [m for m in c["bath"] if m != n]
        Xb, Db, MWb = X[partners], pairs[n, partners], MW[partners]
        first = (1.0 - X[n]) / Xb.sum() * np.sum(Xb / Db)
        second = X[n] * np.sum(Xb * MWb / Db) / np.sum(Xb * MWb)
        assert D[n] == pytest.approx(1.0 / (p * (first + second)), rel=1e-10)


def test_a_bath_species_with_none_of_its_bath_present_is_alone(backend):
    # a bath species' partners are the bath's others: with none of them present, exactly or under the tolerance, it takes its self-diffusion coefficient as a species alone does; every other species' rule over the bath collapses to its pair with the one bath species there
    gas = cases.cantera(ctfile)
    ns, names = gas.n_species, gas.species_names
    mb = cases.kinetic(backend, ctfile, diffusion="binary", bath=["CH4", "O2", "H2O", "CO2"])
    p, T, _ = cases.draw(10, ns)
    pairs = cases.pair_fits(mb.sim.mixture.constants(), T)
    o2 = names.index("O2")
    others = np.arange(ns) != o2
    for trace in (0.0, 1e-44):
        Y = np.zeros(ns)
        Y[o2], Y[names.index("H2")], Y[names.index("H2O")] = 0.5, 0.5, trace
        D = cases.set_state(mb, p, T, Y)[2:]
        assert D[o2] == pytest.approx(pairs[o2, o2] / p, rel=1e-12)
        assert D[others] == pytest.approx(pairs[others, o2] / p, rel=1e-12)


def test_lewis_number(backend):
    species = {name: {**data, "lewis": le} for (name, data), le in zip(two.items(), (1.0, 1.4))}
    mb = cases.solver(backend, species, eos="cpg", trans="constant", diffusion="lewis")
    assert mb.sim.mixture.diffusion.name == "lewis"
    c = mb.sim.mixture.constants()
    assert list(c["lewis"]) == [1.0, 1.4]
    p, T, Y = cases.draw(12, 2)
    mu, kappa, *D = cases.set_state(mb, p, T, Y)
    X = cases.molefrac(Y, c["MW"])
    rho = p * np.sum(X * c["MW"]) / (c["Ru"] * T)
    cp = np.sum(Y * c["cp0"])
    assert kappa == pytest.approx(cases.kappamix(c["kappa0"], X), rel=1e-12)
    assert np.array(D) == pytest.approx(kappa / (rho * cp * c["lewis"]), rel=1e-12)
