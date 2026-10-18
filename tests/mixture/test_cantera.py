from pathlib import Path

import cantera as ct
import numpy as np
import pytest

from peregrine.simulator.mixture.parser import CanteraParser
from peregrine.simulator.mixture import Ru, kB

mechanisms = ("GRI30.yaml", "CH4_O2_FFCMY.yaml", "C2H4_Air_Skeletal.yaml")


def mechPath(name):
    import peregrine.simulator.mixture as mix

    return str(Path(mix.__file__).parent / "database" / "mechanisms" / name)


@pytest.fixture(scope="module", params=mechanisms)
def mech(request):
    path = mechPath(request.param)
    parsed = CanteraParser(path)
    ours, reactions = parsed.species, parsed.reactions
    return ours, reactions, ct.Solution(path), request.param


def test_species_and_reaction_counts(mech):
    ours, reactions, gas, name = mech
    assert list(ours) == list(gas.species_names)
    assert len(reactions) == gas.n_reactions


def test_species_properties(mech):
    ours, _, gas, _ = mech
    for i, name in enumerate(gas.species_names):
        o, s = ours[name], gas.species(i)
        assert o["MW"] == pytest.approx(s.molecular_weight, rel=1e-14)
        # cantera's coeffs are [Tswitch, high a0..a6, low a0..a6]
        c = np.asarray(s.thermo.coeffs)
        n7 = o["NASA7"]
        assert n7["T-range"] == [s.thermo.min_temp, c[0], s.thermo.max_temp]
        assert np.array_equal(np.array(n7["high"]), c[1:8])
        assert np.array_equal(np.array(n7["low"]), c[8:15])
        tr = s.transport
        assert o["well"] / kB == pytest.approx(tr.well_depth / kB, rel=1e-14)
        assert o["diam"] == pytest.approx(tr.diameter, rel=1e-14)
        assert o["dipole"] == pytest.approx(tr.dipole, rel=1e-14, abs=1e-40)
        assert o["polarize"] == pytest.approx(tr.polarizability, rel=1e-14, abs=1e-40)
        assert o["zrot"] == pytest.approx(
            tr.rotational_relaxation, rel=1e-14, abs=1e-40
        )


def test_rate_constants(mech):
    ours, reactions, gas, _ = mech
    for i, o in enumerate(reactions):
        rate = gas.reaction(i).rate
        high = getattr(rate, "high_rate", rate)
        assert o["rate"][0] == pytest.approx(high.pre_exponential_factor, rel=1e-13)
        assert o["rate"][1] == pytest.approx(high.temperature_exponent, rel=1e-13)
        assert o["rate"][2] * Ru == pytest.approx(
            high.activation_energy, rel=1e-13, abs=1e-9
        )
        if "lowrate" in o:
            low = rate.low_rate
            assert o["lowrate"][0] == pytest.approx(
                low.pre_exponential_factor, rel=1e-13
            )
            assert o["lowrate"][1] == pytest.approx(low.temperature_exponent, rel=1e-13)
            assert o["lowrate"][2] * Ru == pytest.approx(
                low.activation_energy, rel=1e-13, abs=1e-9
            )


def test_net_stoichiometry(mech):
    ours, reactions, gas, _ = mech
    names = gas.species_names
    reactant = gas.reactant_stoich_coeffs
    net = gas.product_stoich_coeffs - reactant
    for i, o in enumerate(reactions):
        vec = np.zeros(len(names))
        for sp, v in o["reactants"].items():
            vec[names.index(sp)] -= v
        for sp, v in o["products"].items():
            vec[names.index(sp)] += v
        assert np.allclose(vec, net[:, i]), o["equation"]


# checked through Cantera's rate of progress, not its stoichiometry matrix, which is cancelled and so says nothing about a chaperone on both sides (H + O2 + H2O <=> HO2 + H2O)
def test_forward_rates_of_progress(mech):
    ours, reactions, gas, _ = mech
    gas.TPX = 1400.0, 3.0 * ct.one_atm, {n: 1.0 for n in gas.species_names}
    T, c = gas.T, gas.concentrations
    names = gas.species_names
    rop = gas.forward_rates_of_progress
    for i, o in enumerate(reactions):
        # the falloff and third-body forms carry a pressure dependence this simple evaluation does not reproduce
        if o["type"] != "elementary" or o.get("efficiencies"):
            continue
        A, b, EaOverRu = o["rate"]
        k = A * T**b * np.exp(-EaOverRu / T)
        mine = k * np.prod([c[names.index(s)] ** v for s, v in o["fwd"].items()])
        assert mine == pytest.approx(rop[i], rel=1e-10), o["equation"]


def test_reversibility_and_efficiencies(mech):
    ours, reactions, gas, _ = mech
    for i, o in enumerate(reactions):
        rxn = gas.reaction(i)
        assert o["reversible"] == rxn.reversible, o["equation"]
        theirs = dict(getattr(rxn, "efficiencies", {}) or {})
        if theirs:
            mine = {k: float(v) for k, v in o.get("efficiencies", {}).items()}
            assert mine == pytest.approx(theirs), o["equation"]
