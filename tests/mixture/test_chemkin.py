import cantera as ct
import pytest
from cantera import ck2yaml

import peregrine as pg
from peregrine.simulator.mixture import Mixture, Species, get_mixture
from peregrine.simulator.mixture.parser import CanteraParser, ChemkinParser

species = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "N2", "AR"]

# every reaction kind PEREGRINE has, in the ways CHEMKIN writes them
reactions = """\
REACTIONS CAL/MOLE
H+O2=O+OH                      2.644E+16 -0.6707 17041.00
H+O2+M=HO2+M                   2.800E+18 -0.860      0.00 ! the generic third body
O2/0.00/ H2O/0.00/ N2/0.00/ AR/0.00/
H+2O2=HO2+O2                   3.000E+20 -1.720      0.00
2OH(+M)=H2O2(+M)               7.400E+13 -0.370      0.00
LOW / 2.300E+18 -0.900 -1700.00/
TROE/ 0.7346 94.00 1756.00 5182.00 /
H2/2.00/ H2O/6.00/ AR/0.70/
O+O(+M)=O2(+M)                 1.200E+17 -1.000      0.00
LOW / 1.000E+18 -1.000      0.00/
H+OH(+M)=H2O(+M)               2.200E+22 -2.000      0.00
LOW / 1.000E+20 -1.000      0.00/
TROE/ 0.73 100.0 1500.0 /
O+H(+H2O)=OH(+H2O)             5.000E+17 -1.000      0.00
LOW / 1.000E+18 -1.000      0.00/
HO2+HO2=>H2O2+O2               1.300E+11  0.000  -1630.00
OH+H2O2=HO2+H2O                1.750E+12  0.000    320.00
DUPLICATE
OH+H2O2=HO2+H2O                5.800E+14  0.000   9560.00
 DUP
END
"""


def thermoRecord(sp):
    c = sp.thermo.coeffs  # Tmid, then a1..a7 high, then a1..a7 low
    comp = "".join(f"{el.upper():<2}{n:>3.0f}" for el, n in sp.composition.items())
    head = (
        f"{sp.name:<18}{'':6}{comp:<20}G{sp.thermo.min_temp:10.3f}"
        f"{sp.thermo.max_temp:10.3f}{c[0]:8.2f}{'':6}1"
    )
    numbers = list(c[1:8]) + list(c[8:15])
    rows = [numbers[0:5], numbers[5:10], numbers[10:14]]
    lines = [head]
    for k, row in enumerate(rows):
        lines.append("".join(f"{x:15.8E}" for x in row).ljust(75) + f"{'':4}{k + 2}")
    return lines


def chemkinText(withThermo):
    lines = ["ELEMENTS", "H O N AR", "END", "SPECIES", " ".join(species), "END"]
    if withThermo:
        gas = ct.Solution("gri30.yaml")
        lines += ["THERMO ALL", "   200.000  1000.000  3500.000"]
        for name in species:
            lines += thermoRecord(gas.species(name))
        lines += ["END"]
    return "\n".join(lines) + "\n" + reactions


def assert_close(ours, theirs):
    if isinstance(theirs, dict):
        assert set(ours) == set(theirs)
        for key in theirs:
            assert_close(ours[key], theirs[key])
    elif isinstance(theirs, (list, tuple)):
        assert len(ours) == len(theirs)
        for a, b in zip(ours, theirs):
            assert_close(a, b)
    elif isinstance(theirs, float):
        assert ours == pytest.approx(theirs, rel=1e-12)
    else:
        assert ours == theirs


# ck2yaml needs thermo to convert at all, so Cantera gets the same mechanism with GRI-Mech records attached
@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    where = tmp_path_factory.mktemp("chemkin")
    bare, full, out = where / "mech.inp", where / "full.inp", where / "mech.yaml"
    bare.write_text(chemkinText(False))
    full.write_text(chemkinText(True))
    ck2yaml.convert(str(full), out_name=str(out), quiet=True)
    return str(bare), str(out)


def test_reads_what_cantera_converts(converted):
    ours, theirs = ChemkinParser(converted[0]), CanteraParser(converted[1])
    assert list(ours.species) == list(theirs.species) == species
    assert len(ours.reactions) == len(theirs.reactions) == 10
    for a, b in zip(ours.reactions, theirs.reactions):
        # the equation is written out by each; everything read from it agrees
        assert_close(
            {k: v for k, v in a.items() if k != "equation"},
            {k: v for k, v in b.items() if k != "equation"},
        )


@pytest.mark.parametrize("section", ["THERMO", "TRANSPORT"])
def test_data_sections_are_refused(tmp_path, section):
    inp = tmp_path / "full.inp"
    inp.write_text(
        chemkinText(False).replace("REACTIONS", f"{section}\nEND\nREACTIONS", 1)
    )
    with pytest.raises(ValueError, match=f"{section}.*ck2yaml"):
        ChemkinParser(str(inp))


def test_fortran_exponents():
    assert ChemkinParser._float(" 1.000D+18") == 1e18
    assert ChemkinParser._float("-2.5d-03") == -2.5e-3


def test_species_without_data_come_from_the_library(tmp_path):
    inp = tmp_path / "bare.inp"
    inp.write_text(chemkinText(False))
    parsed = ChemkinParser(str(inp))
    assert all(parsed.species[name] == {} for name in species)
    assert len(parsed.reactions) == 10

    sect = dict(pg.Config()["mixture"])
    sect.update({"T-range": (300.0, 3000.0)}, species=str(inp), eos="tpg", trans="kinetic-theory")
    m = Mixture(sect)
    lib = Species.library()
    assert m.spnames == species
    assert "janaf" in m.species["O2"] and m.species["HO2"]["well"] == lib["HO2"]["well"]
    assert m.species["AR"]["MW"] == pytest.approx(39.95)

    config = pg.Config()
    config["mixture"].update(sect)
    config["mixture"]["chemistry"] = "finite-rate"
    m = get_mixture(config)
    assert int(m.chemistry.rxnconsts(m.species)["nr"]) == 10


@pytest.mark.parametrize(
    "lines, match",
    [
        ("H+O2=O+OH 1.0 0 0\nREV / 1.0 0 0 /", "REV"),
        ("H+O2=O+OH 1.0 0 0\nSRI / 1.0 2.0 3.0 /", "SRI"),
        ("H+XX=O+OH 1.0 0 0", "XX"),
        ("2OH(+M)=H2O2(+M) 1.0 0 0", "LOW"),
        ("H+O2=O+OH 1.0 0 0\nLOW / 1.0 0 0 /", "collider"),
        ("H+O2+M=O+OH 1.0 0 0", "both sides"),
        ("H+O2=O+OH 1.0 0 0\nFOO/1.0/", "FOO"),
    ],
)
def test_what_it_cannot_read_is_refused_by_name(tmp_path, lines, match):
    inp = tmp_path / "bad.inp"
    inp.write_text(
        f"ELEMENTS\nH O\nEND\nSPECIES\n{' '.join(species)}\nEND\nREACTIONS\n{lines}\nEND\n"
    )
    with pytest.raises(ValueError, match=match):
        ChemkinParser(str(inp))
