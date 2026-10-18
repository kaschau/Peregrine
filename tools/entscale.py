import sys
import zipfile

import numpy as np
import yaml

from peregrine.simulator.mixture import Ru, database
from peregrine.simulator.mixture.eos import RealGasModel

# SI with kmol, as the library and the kernels: N_A per kmol
kB, NA = 1.380649e-23, 6.02214076e26
# the global exponents of the two residual functions
muPowers = (1.8, 2.4, 2.8)
kappaPowers = (1.0, 1.5, 2.0, 2.5)
# the paper's fluid names and CoolProp's, by library species
names = {
    "H2": ("HYDROGEN", "Hydrogen"),
    "O2": ("OXYGEN", "Oxygen"),
    "N2": ("NITROGEN", "Nitrogen"),
    "CH4": ("METHANE", "Methane"),
    "H2O": ("WATER", "Water"),
    "CO2": ("CO2", "CarbonDioxide"),
    "CO": ("CO", "CarbonMonoxide"),
    "AR": ("ARGON", "Argon"),
    "C2H6": ("ETHANE", "Ethane"),
    "C2H4": ("ETHYLENE", "Ethylene"),
    "C2H2": ("ACETYLENE", "Acetylene"),
    "C3H8": ("PROPANE", "n-Propane"),
    "C3H6": ("PROPYLEN", "Propylene"),
    "NH3": ("AMMONIA", "Ammonia"),
    "CH3OH": ("METHANOL", "Methanol"),
    "N2O": ("N2O", "NitrousOxide"),
    "SF6": ("SF6", "SulfurHexafluoride"),
    "NO": (None, "NitricOxide"),
}
citation = "Yang, Bell et al., ACS Omega 2025, 10, 6124, SI, SRK fit"


def paper_tables(path):
    folder = "Supporting Information/Parameters/"
    out = {}
    with zipfile.ZipFile(path) as z:
        for key, member in (
            ("mu", "RES parameters for viscosity/RES_V_Parameters_SRK.txt"),
            ("kappa", "RES parameters for thermal conductivity/RES_TC_Parameters_SRK.txt"),
        ):
            table = {}
            for line in z.read(folder + member).decode().splitlines()[1:]:
                f = line.split()
                if len(f) < 13:
                    continue
                table[f[0]] = dict(
                    group=int(f[2]),
                    glb=[float(x) for x in f[3:7]],
                    xi=float(f[7]),
                    fitted=int(f[8]) == 1,
                    ind=[float(x) for x in f[9:13]],
                )
            out[key] = table
    return out


# every fluid of a group carries the same global parameters, so any row of the group serves
def groupRows(tables):
    return {
        key: {row["group"]: row["glb"] for row in table.values()}
        for key, table in tables.items()
    }


def cubic_volume_splus(T, p, Tc, pc, w):
    c = RealGasModel.cubic
    m = c["m"][0] + c["m"][1] * w + c["m"][2] * w**2
    ac = c["omega_a"] * (Ru * Tc) ** 2 / pc
    b = c["omega_b"] * Ru * Tc / pc
    base = 1.0 + m * (1.0 - np.sqrt(T / Tc))
    a = ac * base**2
    da = -ac * m * abs(base) / np.sqrt(T * Tc)
    u, wf = c["u"], c["w"]
    roots = np.roots([p, -Ru * T + p * b * (u - 1.0), a - u * b * Ru * T + p * b * b * (wf - u), -a * b - wf * b * b * Ru * T - wf * p * b**3])
    v = roots[np.abs(roots.imag) < 1e-9 * np.abs(roots).max()].real
    # below Tc the smallest real root above the covolume is the liquid root, the one wanted
    v = v[v > b].min()
    splus = -(np.log(1.0 - b / v) + da / (b * Ru) * np.log(1.0 + b / v))
    return v, splus


def fitViscosity(cpName, Tc, pc, w):
    import CoolProp.CoolProp as CP

    Tmin, Tmax, pmax = (CP.PropsSI(k, cpName) for k in ("Tmin", "Tmax", "pmax"))
    # kg/kmol, so MW / NA is the mass of one molecule
    MW = CP.PropsSI("M", cpName) * 1e3
    A, y = [], []
    # no sample near the critical point
    for Tr in (0.6, 0.7, 0.8, 1.2, 1.5, 2.0, 3.0, 5.0):
        T = Tr * Tc
        if not Tmin <= T <= Tmax:
            continue
        eta0 = CP.PropsSI("V", "T", T, "Dmolar", 1e-6, cpName)
        for p in (10e6, 20e6, 30e6, 45e6, 60e6, 80e6):
            if p > pmax:
                continue
            eta = CP.PropsSI("V", "T", T, "P", p, cpName)
            v, s = cubic_volume_splus(T, p, Tc, pc, w)
            rhoN, m = NA / v, MW / NA
            etaPlus = (eta - eta0) * s ** (2.0 / 3.0) / (rhoN ** (2.0 / 3.0) * np.sqrt(m * kB * T))
            if etaPlus > 0.0:
                A.append([s**q for q in muPowers])
                y.append(np.log(etaPlus + 1.0))
    return np.linalg.lstsq(np.array(A), np.array(y), rcond=None)[0]


def hasCoolPropTransport(cpName):
    import CoolProp.CoolProp as CP

    try:
        CP.PropsSI("V", "T", 300.0, "P", 1e5, cpName)
        CP.PropsSI("L", "T", 300.0, "P", 1e5, cpName)
        return True
    except Exception:
        return False


# the paper's group 3 is light hydrocarbons, group 2 the gases (N2, O2, Ar, CH4)
def predictedGroup(comp):
    return 3 if comp.get("C", 0) > 0 and comp.get("H", 0) > 0 and comp.get("N", 0) == 0 else 2


def fmt(row):
    return "[" + ", ".join(f"{x:.6f}" for x in row) + "]"


# yaml reads a bare NO as a boolean, so a name it would misread is quoted
def yamlkey(name):
    return name if yaml.safe_load(f"{name}:") == {name: None} else f'"{name}"'


# zipPath is the paper's Supporting Information zip, figshare article 28376075 (CC BY-NC 4.0)
def main(zipPath):
    tables = paper_tables(zipPath)
    groups = groupRows(tables)
    library = database("species")
    for name, entry in library.items():
        si, cp = names.get(name, (None, None))
        coefficients, notes = {}, {}
        for key, terms in (("mu", 3), ("kappa", 4)):
            row = tables[key].get(si) if si else None
            if row is not None and row["fitted"]:
                coefficients[key], notes[key] = row["ind"][:terms], citation
            elif key == "mu" and cp and hasCoolPropTransport(cp):
                coefficients[key] = list(fitViscosity(cp, entry["Tcrit"], entry["pcrit"], entry["acentric"]))
                notes[key] = f"fitted here on CoolProp's {cp} reference viscosity with PEREGRINE's cubic, the paper's form"
            else:
                g = predictedGroup(entry["comp"])
                coefficients[key] = groups[key][g][:terms]
                notes[key] = f"predicted: group {g} parameters of Yang et al., scaling factor 1"
        print(f"{yamlkey(name)}:")
        if notes["mu"] == notes["kappa"]:
            print(f"  # residual entropy scaling: {notes['mu']}")
        else:
            print(f"  # residual entropy scaling: resmu {notes['mu']}; reskappa {notes['kappa']}")
        print(f"  resmu: {fmt(coefficients['mu'])}")
        print(f"  reskappa: {fmt(coefficients['kappa'])}")


if __name__ == "__main__":
    main(sys.argv[1])
