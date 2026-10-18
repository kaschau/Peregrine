import pytest

from peregrine.simulator.mixture import Mixture, Ru, Species, kB
from peregrine.simulator.mixture.eos import RealGasModel
from peregrine.simulator.mixture.transport import ChungDenseGasModel

from tests.eos.support import composition, config, cubicvolume, setstate, solver, state, textbookdensity

mechanism = "CH4_O2_FFCMY.yaml"
derived = ("O", "HO2", "CH3")


def section(species):
    return config(species, "real-gas", (300.0, 4000.0))["mixture"]


@pytest.mark.parametrize("name", ["O2", "N2", "CH4", "CO2", "CO", "C2H4", "NH3"])
def test_the_fallback_lands_near_the_measured_point(name):
    # not AR, H or OH, whose library point is the fallback already, nor H2 or H2O, which the correlation misses by half; a unit slip misses by orders
    data = Species.library()[name]
    broken = Species(name, {k: v for k, v in data.items() if k not in RealGasModel.critical})
    model = RealGasModel(section([name]))
    Tc, pc, Vc, w = (got[0] for got in model.critpoint({name: broken}))
    assert Tc == pytest.approx(data["Tcrit"], rel=0.2)
    assert pc == pytest.approx(data["pcrit"], rel=0.2)
    assert Vc == pytest.approx(data["Vcrit"], rel=0.1)
    assert w == 0.0


def test_a_mechanism_keeps_measured_points_and_derives_the_rest():
    sect = dict(section(mechanism), trans="chung")
    m = Mixture(sect)
    library = Species.library()
    for name in m.spnames:
        sp, lib = m.species[name], library[name]
        if name in derived:
            assert "Tcrit" not in lib
            # Tee, Gotoh and Stewart's relations run backwards, Pitzer's Zc at zero acentric factor
            Tc = 1.2593 * sp["well"] / kB
            Vcmol = (sp["diam"] * 1e10 / 0.809) ** 3 * 1e-3
            assert sp["Tcrit"] == pytest.approx(Tc, rel=1e-12)
            assert sp["pcrit"] == pytest.approx(0.291 * Ru * Tc / Vcmol, rel=1e-12)
            assert sp["Vcrit"] == pytest.approx(Vcmol / sp["MW"], rel=1e-12)
            assert sp["acentric"] == 0.0
        else:
            for key in RealGasModel.critical:
                assert sp[key] == lib[key]
    # the regression numbers of the fallback
    assert m.species["CH3"]["Tcrit"] == pytest.approx(181.339, rel=1e-3)
    assert m.species["HO2"]["Tcrit"] == pytest.approx(135.249, rel=1e-3)
    assert m.species["O"]["Tcrit"] == pytest.approx(100.744, rel=1e-3)
    # Chung's rules give a derived species its well and diameter back
    sigma, epsk = ChungDenseGasModel(sect).chungparams(m.species)
    for name in derived:
        n = m.spnames.index(name)
        assert sigma[n] == pytest.approx(m.species[name]["diam"] * 1e10, rel=1e-12)
        assert epsk[n] == pytest.approx(m.species[name]["well"] / kB, rel=1e-12)


@pytest.fixture(scope="module")
def mb(backend):
    return solver(backend, mechanism, "real-gas", (300.0, 4000.0))


def test_the_cubic_runs_on_the_derived_points(mb):
    p, T = 60e6, 300.0
    Y = composition(mb, CH4=0.2, O2=0.77, O=0.01, HO2=0.01, CH3=0.01)
    sp = [mb.sim.mixture.species[n] for n in mb.sim.mixture.spnames]
    setstate(mb, p, T, Y)
    assert state(mb)["rho"] == pytest.approx(textbookdensity(sp, p, T, Y), rel=1e-10)
    # and a pure radical is the cubic on its derived point alone
    sp = mb.sim.mixture.species["CH3"]
    setstate(mb, p, T, composition(mb, CH3=1.0))
    v = cubicvolume(p, T, sp["Tcrit"], sp["pcrit"], sp["acentric"])
    assert state(mb)["rho"] == pytest.approx(sp["MW"] / v, rel=1e-10)
