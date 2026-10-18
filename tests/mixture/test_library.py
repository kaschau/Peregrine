from pathlib import Path

import CoolProp
import CoolProp.CoolProp as CP
import cantera as ct
import numpy as np
import pytest
import yaml

from peregrine.simulator.mixture import Ru, Species
from peregrine.simulator.mixture.thermo import BaseThermoData

library = Species.library()
gri = ct.Solution("gri30.yaml")
shared = [name for name in library if name in gri.species_names]

# library less GRI-Mech 3.0, MJ/kmol: JANAF's formation enthalpies for these radicals are not GRI-Mech's
dhf_known = {
    "C2H": -89.226,
    "NCO": 27.614,
    "NH": 19.644,
    "HNCO": 16.407,
    "HO2": -10.460,
    "CH2O": -7.318,
    "HNO": -6.679,
    "CH2": -5.941,
    "HCN": 4.328,
    "CN": -3.545,
    "CH": -3.207,
}
# library over GRI-Mech 3.0 less one, at 298.15 K
s298_known = {"C2H": -2.76e-2, "HNCO": -1.04e-2}
# cp, relative over the table's range, where JANAF and GRI-Mech's polynomial part ways (measured 0.12, 0.14, 0.05)
cp_loose = {"C2H": 0.2, "HNO": 0.2, "H2O2": 0.1}
# the transport fields the library states otherwise than GRI-Mech: corresponding states from the critical point, dipoles and polarizabilities GRI-Mech leaves at zero, and the molecule's shape
transport_choices = {
    "H": {"polarize"},
    "O": {"polarize"},
    "N": {"polarize"},
    "C": {"polarize"},
    "OH": {"dipole"},
    "CH": {"dipole"},
    "HNO": {"dipole"},
    "CN": {"dipole"},
    "HCNO": {"dipole"},
    "HNCO": {"dipole"},
    "NCO": {"dipole"},
    "CH2O": {"dipole", "polarize"},
    "CH2CO": {"dipole", "polarize"},
    "NO": {"dipole", "polarize"},
    "NO2": {"dipole", "polarize"},
    "HCN": {"dipole", "polarize"},
    "CH3CHO": {"dipole", "polarize"},
    "H2O2": {"well", "diam", "dipole"},
    "CH3OH": {"well", "diam", "dipole", "polarize"},
    "NH3": {"well", "diam", "dipole", "polarize"},
    "N2O": {"well", "diam", "dipole", "polarize"},
    "C3H8": {"well", "diam", "dipole", "polarize"},
    "C2H2": {"well", "diam", "polarize"},
    "C2H4": {"well", "diam", "polarize"},
    "C2H6": {"well", "diam", "polarize"},
    "AR": {"well", "diam", "polarize"},
    "CH2": {"geometry"},
    "CH2(S)": {"geometry"},
    "CH3": {"geometry"},
    "HCCO": {"geometry"},
    "NNH": {"geometry"},
    "H2CN": {"geometry"},
    "HCNN": {"geometry"},
}
# the species the real-gas cases run on, and CoolProp's fluid of each
fluids = {
    "O2": "Oxygen",
    "N2": "Nitrogen",
    "CH4": "Methane",
    "CO2": "CarbonDioxide",
    "H2O": "Water",
    "H2": "Hydrogen",
    "CO": "CarbonMonoxide",
    "AR": "Argon",
}


def thermo(name):
    sp = Species(name, dict(library[name]))
    fmt = BaseThermoData.find(sp)
    T, cpR = fmt.cp_r(sp)
    hR, sR = fmt.hsref_r(sp)
    return T, cpR * Ru, hR * Ru, sR * Ru


@pytest.mark.parametrize("name", shared)
def test_formation_enthalpies_and_entropies_are_gri_mechs(name):
    _, _, h, s = thermo(name)
    th = gri.species(name).thermo
    gap = (h - th.h(298.15)) / 1e6
    if name in dhf_known:
        assert gap == pytest.approx(dhf_known[name], abs=0.5)
    else:
        assert abs(gap) < 2.0
    gap = s / th.s(298.15) - 1.0
    if name in s298_known:
        assert gap == pytest.approx(s298_known[name], abs=2e-3)
    else:
        assert abs(gap) < 1e-2


@pytest.mark.parametrize("name", shared)
def test_cp_follows_gri_mechs_over_the_table(name):
    # looser past 1500 K, where GRI-Mech's polynomials themselves stray (methane's passes the classical limit by 3000 K)
    T, cp, _, _ = thermo(name)
    th = gri.species(name).thermo
    inside = (T >= 300.0) & (T <= min(T[-1], 3000.0))
    theirs = np.array([th.cp(t) for t in T[inside]])
    error = np.abs(cp[inside] / theirs - 1.0)
    bound = cp_loose.get(name, np.where(T[inside] <= 1500.0, 0.05, 0.15))
    assert np.all(error < bound), (name, T[inside][np.argmax(error - bound)])


@pytest.mark.parametrize("name", [n for n in shared if "NASA7" in library[n]])
def test_nasa7_entries_are_gri_mechs_coefficients(name):
    n7 = library[name]["NASA7"]
    c = gri.species(name).thermo.coeffs
    assert n7["T-range"][1] == c[0]
    assert np.array_equal(np.array(n7["high"]), c[1:8])
    assert np.array_equal(np.array(n7["low"]), c[8:15])


@pytest.mark.parametrize("name", shared)
def test_transport_fields_are_gri_mechs_unless_chosen_otherwise(name):
    tr = gri.species(name).transport
    theirs = dict(well=tr.well_depth, diam=tr.diameter, dipole=tr.dipole, polarize=tr.polarizability, zrot=tr.rotational_relaxation, geometry=tr.geometry)
    chosen = transport_choices.get(name, set())
    for field, value in theirs.items():
        ours = library[name][field]
        same = ours == value if isinstance(value, str) else ours == pytest.approx(value, rel=1e-6, abs=1e-40)
        # the list of choices is exact: a field on it differs, one off it does not
        assert same == (field not in chosen), (name, field, ours, value)


def test_the_measured_critical_points_are_canteras():
    table = yaml.safe_load(open(Path(ct.__file__).parent / "data" / "critical-properties.yaml"))
    theirs = {entry["name"]: entry for entry in table["species"]}
    checked = 0
    for name, data in library.items():
        if "Tcrit" not in data or name not in theirs:
            continue
        crit = theirs[name]["critical-parameters"]
        assert data["Tcrit"] == float(crit["critical-temperature"])
        assert data["pcrit"] == float(crit["critical-pressure"])
        assert data["acentric"] == float(crit["acentric-factor"])
        if "critical-molar-volume" in crit:
            MW = Species.molwt(data["comp"]) if "comp" in data else data["MW"]
            assert data["Vcrit"] * MW == pytest.approx(float(crit["critical-molar-volume"]), rel=1e-6)
        checked += 1
    assert checked == 12


@pytest.mark.parametrize("name", list(fluids))
def test_the_real_gas_species_cp_is_the_reference_equations(name):
    AS = CoolProp.AbstractState("HEOS", fluids[name])
    T, cp, _, _ = thermo(name)
    # JANAF's tables sit within 1e-3 of the reference equations' ideal part but methane's, 5e-3 from Setzmann and Wagner's spectroscopic one at 600 K
    within = 1e-2 if name == "CH4" else 2e-3
    for t in np.arange(300.0, min(1500.0, AS.Tmax(), T[-1]) + 1.0, 100.0):
        AS.update(CP.PT_INPUTS, 1e3, t)
        theirs = AS.cp0mass() * AS.molar_mass() * 1e3
        assert np.interp(t, T, cp) == pytest.approx(theirs, rel=within), (name, t)
