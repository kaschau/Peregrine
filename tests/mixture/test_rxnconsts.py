from collections import Counter
from pathlib import Path

import cantera as ct
import numpy as np
import pytest

import peregrine as pg
from peregrine.misc import subclass_where
from peregrine.simulator.mixture import Ru, get_mixture
from peregrine.simulator.mixture.chemistry import FiniteRateModel
from peregrine.simulator.mixture.reaction import BaseReaction

from tests.mixture.rates import density, oneway_twin, production, texp, tlog, troefiterr

mechanisms = ("GRI30.yaml", "CH4_O2_FFCMY.yaml", "C2H4_Air_Skeletal.yaml")
# the refit thermo's consequence on the reverse rates, of the largest rate (measured 5e-6, 4e-4, 4e-4)
reversible_within = {"GRI30.yaml": 2e-3, "CH4_O2_FFCMY.yaml": 1e-4, "C2H4_Air_Skeletal.yaml": 2e-3}
T_range = (300.0, 3500.0)


def mechpath(name):
    import peregrine.simulator.mixture as mix

    return str(Path(mix.__file__).parent / "database" / "mechanisms" / name)


def config(name, root=None):
    config = pg.Config()
    config["mixture"].update({"species": name, "eos": "tpg", "T-range": T_range, "chemistry": "finite-rate"})
    return config


@pytest.fixture(scope="module", params=mechanisms)
def mech(request):
    name = request.param
    gas = ct.Solution(mechpath(name))
    mixture = get_mixture(config(name))
    return mixture, mixture.chemistry.rxnconsts(mixture.species), gas, name


def states(gas):
    for T, p in ((800.0, 1e5), (1500.0, 3e5), (2800.0, 30e5)):
        gas.TPX = T, p, {n: 1.0 for n in gas.species_names}
        yield T, gas.concentrations


# Cantera's convention: the third body's concentration is left out of a three-body reaction's constant and kept in a falloff's
def rateconstant(r, row, T, c):
    lnT = np.log(T)
    kinf = np.exp(r["logA"][row] + r["b"][row] * lnT - r["EaR"][row] / T)
    kind = subclass_where(BaseReaction, name=r["type"][row])
    if not kind.falloff:
        return kinf
    meff = r["defeff"][row] * c.sum() + np.sum(r["effdev"][row] * c[r["effsp"][row]])
    k0 = kinf * np.exp(r["logprA"][row] + r["prb"][row] * lnT - r["prEaR"][row] / T)
    pr = k0 * meff / kinf
    F = 1.0
    if kind.troe:
        terms = r["nfcent"][row]
        fcent = np.sum(r["fcentsign"][row, :terms] * np.exp(r["fcentc0"][row, :terms] + r["fcentcT"][row, :terms] * T + r["fcentcTinv"][row, :terms] / T))
        logfc = np.log10(fcent)
        C, N = -0.4 - 0.67 * logfc, 0.75 - 1.27 * logfc
        f1 = (np.log10(pr) + C) / (N - 0.14 * (np.log10(pr) + C))
        F = 10 ** (logfc / (1 + f1 * f1))
    return kinf * pr / (1 + pr) * F


def test_the_rows_hold_every_reaction_of_each_kind(mech):
    mixture, r, gas, name = mech
    assert int(r["nr"]) == gas.n_reactions == len(mixture.reactions)
    counted = Counter(r["type"].tolist())
    assert sum(counted.values()) == gas.n_reactions and min(counted[k] for k in ("three-body", "lindemann", "troe")) > 0
    assert r["fwdsp"].dtype == r["netsp"].dtype == r["revsp"].dtype == r["effsp"].dtype == np.int32


def test_every_forward_rate_constant_is_canteras(mech):
    mixture, r, gas, name = mech
    for T, c in states(gas):
        gas.TP = T, gas.P
        theirs = gas.forward_rate_constants
        for row in range(int(r["nr"])):
            assert rateconstant(r, row, T, c) == pytest.approx(theirs[row], rel=1e-12), mixture.reactions[row]["equation"]
            kind = subclass_where(BaseReaction, name=r["type"][row])
            if kind.thirdbody and not kind.falloff:
                meff = r["defeff"][row] * c.sum() + np.sum(r["effdev"][row] * c[r["effsp"][row]])
                assert meff == pytest.approx(gas.third_body_concentrations[row], rel=1e-12)


def test_the_reverse_orders_are_the_products_coefficients(mech):
    mixture, r, gas, name = mech
    index = {n: k for k, n in enumerate(mixture.spnames)}
    for row, rxn in enumerate(mixture.reactions):
        baked = {int(k): v for k, v in zip(r["revsp"][row], r["revorder"][row]) if v != 0}
        products = {index[sp]: v for sp, v in rxn["products"].items()}
        assert baked == (products if rxn["reversible"] else {})


def test_the_table_log_is_the_librarys_to_rounding():
    # 3e-14 at a concentration's magnitudes, which the affinity amplifies to 1e-10 of the largest rate near equilibrium on GRI-3.0, inside the kernel-against-twin bound
    x = np.geomspace(1e-300, 1e300, 200001)
    exact = np.log(x)
    assert np.all(np.abs(tlog(x) - exact) < 1.5e-13 + 4.0 * np.spacing(np.abs(exact)))
    assert abs(tlog(np.finfo(np.float64).tiny) - np.log(np.finfo(np.float64).tiny)) < 2e-13
    # nothing, and a denormal, land far below any floor
    assert tlog(np.array([0.0, 1e-310])).max() < -700.0


def test_the_table_exp_is_the_librarys_to_rounding():
    x = np.linspace(-708.0, 709.0, 200001)
    exact = np.exp(x)
    assert np.all(np.abs(texp(x) - exact) <= 2.0 * np.spacing(exact))
    # below the normal range a tiny positive number, never zero
    # the floor at the smallest normal exponent: below -708.4 a number near 2e-308, never zero
    low = texp(np.linspace(-745.0, -708.5, 2001))
    assert np.all((low >= 2.2e-308) & (low <= 4.5e-308))


def test_troes_centre_fit_is_within_the_parameters_precision(mech):
    # 1e-3 in log10 Fcent is 2.3e-3 in the rate at the falloff's centre, inside the two or three figures Troe parameters are given to
    assert troefiterr(mech[0]) < 2.303 * 1e-3


def test_troes_centre_broadening_is_the_formula(mech):
    mixture, r, gas, name = mech
    T = np.geomspace(100.0, 20000.0, 400)
    seen = 0
    for rxn in mixture.reactions:
        if "troe" not in rxn:
            continue
        a, T3, T1, T2 = rxn["troe"]
        theirs = (1 - a) * np.exp(-T / T3) + a * np.exp(-T / T1) + np.exp(-T2 / T)
        terms = FiniteRateModel._fcent(rxn["troe"])
        ours = sum(s * np.exp(c0 + cT * T + cTi / T) for s, c0, cT, cTi in terms)
        assert np.abs(ours / theirs - 1.0).max() < 1e-14
        seen += 1
    assert seen > 0


def test_a_sentinel_drops_the_term_it_disables():
    # no shipped mechanism carries a sentinel: T3 below rounding drops its term, T1 past rounding makes its term the constant a, a zero T2 drops the third
    T = np.array([100.0, 1000.0, 20000.0])

    def fcent(troe):
        terms = FiniteRateModel._fcent(troe)
        return terms, sum(s * np.exp(c0 + cT * T + cTi / T) for s, c0, cT, cTi in terms)

    terms, value = fcent([0.6, 1e-30, 1e30, 0.0])
    assert len(terms) == 1 and np.allclose(value, 0.6, rtol=1e-15)
    terms, value = fcent([0.5, 100.0, 2000.0, 0.0])
    assert len(terms) == 2 and np.allclose(value, 0.5 * np.exp(-T / 100.0) + 0.5 * np.exp(-T / 2000.0), rtol=1e-15)
    terms, value = fcent([1.0, 100.0, 2000.0, 5000.0])
    assert len(terms) == 2 and np.allclose(value, np.exp(-T / 2000.0) + np.exp(-5000.0 / T), rtol=1e-15)
    terms, value = fcent([0.0, 100.0, 2000.0, 5000.0])
    assert len(terms) == 2 and np.allclose(value, np.exp(-T / 100.0) + np.exp(-5000.0 / T), rtol=1e-15)
    assert FiniteRateModel._fcent(None) == []


def test_every_reaction_conserves_mass(mech):
    mixture, r, gas, name = mech
    rows = r["netnuMW"]
    assert np.all(np.abs(rows.sum(axis=1)) < 1e-12 * np.abs(rows).max(axis=1))


def test_the_floor_suppresses_an_absent_species(mech):
    # one reactant at the floor and every other at e^12 kmol/m3 keeps every direction's log rate below -45, the kernel's extra for a third body and a falloff counted
    mixture, r, gas, name = mech
    floor = float(r["logcfloor"])
    assert floor <= -90.0
    T = np.geomspace(*T_range, 256)
    logT, gRT = np.log(T), mixture.eos.gibbs(mixture.species, T)
    index = {n: k for k, n in enumerate(mixture.spnames)}
    worst = -np.inf
    for rxn in mixture.reactions:
        A, b, EaR = rxn["rate"]
        logkf = np.log(A) + b * logT - EaR / T
        aux = subclass_where(BaseReaction, name=rxn["type"]).boost(12.0)
        fwd = [(index[sp], v) for sp, v in rxn["fwd"].items() if v > 0]
        for k, vk in fwd:
            other = sum(12.0 * v for n, v in fwd if n != k)
            worst = max(worst, logkf.max() + aux + other + vk * floor)
        if not rxn["reversible"]:
            continue
        net = {}
        for sp, v in rxn["reactants"].items():
            net[index[sp]] = net.get(index[sp], 0.0) - v
        for sp, v in rxn["products"].items():
            net[index[sp]] = net.get(index[sp], 0.0) + v
        affinity = sum(v * gRT[n] for n, v in net.items())
        logkr = logkf + affinity + sum(net.values()) * np.log(Ru * T / 101325.0)
        rev = [(index[sp], v) for sp, v in rxn["products"].items() if v > 0]
        for k, vk in rev:
            other = sum(12.0 * v for n, v in rev if n != k)
            worst = max(worst, logkr.max() + aux + other + vk * floor)
    # the worst reaction sits on the margin exactly, which is how the floor is set
    assert worst < -45.0 + 1e-9
    assert worst > -46.0


def test_the_product_form_takes_every_reaction_inside_the_range(mech):
    # the guard: whole orders at the stoichiometry, every exponential's argument and every chain of factors inside the table exp's range over the T-range
    mixture, r, gas, name = mech
    assert np.all(r["product"] == 1)
    kinds = {mixture.chemistry.terms(mixture.species).reactions[k].name for k in range(int(r["nr"]))}
    assert {"elementary", "three-body"} <= kinds
    # a lower bound at 100 K puts a reaction's K below the floor for some of GRI-3.0's, which keep the log form
    cold = get_mixture(config(name))
    cold.cfgsect["T-range"] = (100.0, 5000.0)
    rc = cold.chemistry.rxnconsts(cold.species)
    assert rc["product"].sum() <= r["product"].sum()
    if name == "GRI30.yaml":
        assert 0 < rc["product"].sum() < r["product"].sum()


def test_the_two_forms_agree_to_rounding(mech):
    # the log form against the product form on the same states, each as the kernel evaluates it: the exponent's rounding at the forms' extremes
    mixture, r, gas, name = mech
    for T, c in states(gas):
        rho, Y = c.sum() * gas.mean_molecular_weight, gas.Y
        log, product = production(mixture, rho, Y, T, form="log"), production(mixture, rho, Y, T, form="product")
        assert np.abs(log - product).max() < 3e-14 * np.abs(log).max()


def test_the_production_rates_compose_to_canteras(mech, tmp_path):
    # the one-way twin carries no refit thermo in a reverse rate, so each reaction body is checked at rounding (measured 2e-14); then the mechanism itself
    mixture, r, gas, name = mech
    twin = oneway_twin(gas, name, tmp_path)
    oneway = get_mixture(config(name), root=str(tmp_path))
    assert int(oneway.chemistry.rxnconsts(oneway.species)["reversible"].sum()) == 0
    rng = np.random.default_rng(7)
    for trial in range(3):
        Y = rng.random(mixture.ns)
        Y /= Y.sum()
        p, T = rng.uniform(1e5, 30e5), rng.uniform(600.0, 2800.0)
        rho = density(mixture, Y, T, p)
        twin.TPY = T, p, Y
        theirs = twin.net_production_rates * twin.molecular_weights
        # the formulation with Troe's centres exact, then as the kernel fits them, within the fit's budget
        ours = production(oneway, rho, Y, T, exact=True)
        assert np.abs(ours - theirs).max() < 1e-10 * np.abs(theirs).max()
        assert np.all(np.abs(ours - theirs) <= 1e-10 * np.abs(theirs) + 1e-20 * np.abs(theirs).max())
        fiterr = troefiterr(oneway)
        fitted = production(oneway, rho, Y, T)
        assert np.all(np.abs(fitted - theirs) <= 1e-10 * np.abs(theirs) + fiterr * np.abs(theirs).max())
        gas.TPY = T, p, Y
        theirs = gas.net_production_rates * gas.molecular_weights
        ours = production(mixture, rho, Y, T)
        scale = np.abs(theirs).max()
        assert np.abs(ours - theirs).max() < (reversible_within[name] + fiterr) * scale
        assert abs(ours.sum()) < 1e-12 * scale
