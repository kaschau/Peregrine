from pathlib import Path

import cantera as ct
import numpy as np
import pytest

from tests.eos.support import WithEosKernels, cell, setstate, solver, state

here = Path(__file__).parent
cpgfile = str(here / "ct_test_cpg.yaml")
mechanisms = str(here / "../../peregrine/simulator/mixture/database/mechanisms")
ct.add_directory(mechanisms)


def dense_states():
    return [
        (59.8e6, 130.0, np.array([0.5, 0.3, 0.2])),
        (59.8e6, 111.7, np.array([0.02, 0.9, 0.08])),
        (59.8e6, 300.0, np.array([0.3, 0.3, 0.4])),
        (5e6, 400.0, np.array([0.2, 0.5, 0.3])),
    ]


def ideal_states(ns):
    rng = np.random.default_rng(3)
    states = []
    for p, T in ((1e5, 300.0), (5e5, 800.0), (5e5, 2500.0)):
        Y = rng.random(ns)
        states.append((p, T, Y / Y.sum()))
    return states


@pytest.fixture(scope="module")
def cpg(backend):
    return solver(backend, cpgfile, "cpg", cls=WithEosKernels)


@pytest.fixture(scope="module")
def tpg(backend):
    return solver(backend, "CH4_O2_FFCMY.yaml", "tpg", cls=WithEosKernels)


@pytest.fixture(scope="module")
def realgas(backend):
    return solver(backend, ["CH4", "O2", "N2"], "real-gas", (100.0, 1500.0), cls=WithEosKernels)


def case(request, eos):
    mb = request.getfixturevalue({"cpg": "cpg", "tpg": "tpg", "real-gas": "realgas"}[eos])
    states = dense_states() if eos == "real-gas" else ideal_states(mb.sim.mixture.ns)
    return mb, states


def enthalpies_at(mb, p, T, Y):
    setstate(mb, p, T, Y)
    mb.launch(mb.kernels["hpartial"](Q="Q"), "interior")
    s = state(mb)
    return s["rho"], s["rho"] * s["e"], s["h"], cell(mb, "hi")


@pytest.mark.parametrize("eos", ["cpg", "tpg", "real-gas"])
def test_partial_enthalpies_sum_to_the_mixtures(request, eos):
    mb, states = case(request, eos)
    for p, T, Y in states:
        _, _, H, hi = enthalpies_at(mb, p, T, Y)
        assert np.dot(Y, hi) == pytest.approx(H, rel=1e-10)


@pytest.mark.parametrize("eos", ["cpg", "tpg", "real-gas"])
def test_energys_derivative_by_a_mass_fraction_is_the_eoss(request, eos):
    # the floor is the state's rounding through the difference, measured 3e-6 for tpg where two species' enthalpies nearly cancel
    mb, states = case(request, eos)
    for p, T, Y in states:
        for n in range(min(2, len(Y) - 1)):
            d = 1e-5
            up, down = Y.copy(), Y.copy()
            up[n] += d
            up[-1] -= d
            down[n] -= d
            down[-1] += d
            rhoup, Eup, _, _ = enthalpies_at(mb, p, T, up)
            rhodown, Edown, _, _ = enthalpies_at(mb, p, T, down)
            rho, _, H, hi = enthalpies_at(mb, p, T, Y)
            rho_Y = (rhoup - rhodown) / (2 * d)
            dEdY = (Eup - Edown) / (2 * d)
            assert rho_Y * H + rho * (hi[n] - hi[-1]) == pytest.approx(dEdY, rel=1e-5)


def test_the_calorically_perfect_species_enthalpies_are_canteras(cpg):
    # the file's species have cp0 alone, referenced at 0 K: h_n = cp0_n T, exactly
    gas = ct.Solution(cpgfile)
    for p, T, Y in ideal_states(gas.n_species):
        gas.TPY = T, p, Y
        _, _, _, hi = enthalpies_at(cpg, p, T, Y)
        theirs = gas.partial_molar_enthalpies / gas.molecular_weights
        assert np.abs(hi / theirs - 1.0).max() < 1e-12


def test_the_thermally_perfect_species_enthalpies_are_canteras(tpg):
    # within the refit, judged against the sensible enthalpy cp T as test_tpg does (measured 4e-4)
    gas = ct.Solution("CH4_O2_FFCMY.yaml")
    achieved = max(sp["cpfiterr"] for sp in tpg.sim.mixture.species.values()) + 1e-8
    for p, T, Y in ideal_states(gas.n_species):
        gas.TPY = T, p, Y
        _, _, _, hi = enthalpies_at(tpg, p, T, Y)
        theirs = gas.partial_molar_enthalpies / gas.molecular_weights
        assert np.abs(hi - theirs).max() / (gas.cp * T) < achieved
