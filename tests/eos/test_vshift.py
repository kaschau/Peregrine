import cantera as ct
import numpy as np
import pytest

from tests.eos.support import composition, cubicvolume, set_tguess, setstate, solver, state


def roundtrip(mb, p, T, Y, guessoff=0.0):
    setstate(mb, p, T, Y)
    rho = state(mb)["rho"]
    set_tguess(mb, by=guessoff)
    mb.postrhs()
    back = state(mb)
    return rho, back["p"], back["T"]


@pytest.fixture(scope="module")
def shifted(backend):
    return solver(backend, {"O2": {"vshift": 1e-3}, "CH4": {}, "N2": {}, "CO2": {}}, "real-gas", (100.0, 1000.0))


@pytest.mark.parametrize(
    "name, fluid, temperatures",
    [
        ("O2", ct.Oxygen, (111.73, 154.58, 300.0, 784.2)),
        ("CH4", ct.Methane, (162.69, 220.0, 500.0)),
    ],
)
def test_density_follows_the_pure_fluid(dense, name, fluid, temperatures):
    fluid = fluid()
    Y = composition(dense, **{name: 1.0})
    for p in (40e6, 59.761e6, 80e6):
        for T in temperatures:
            fluid.TP = T, p
            rho, _, _ = roundtrip(dense, p, T, Y)
            # at combustor pressures Soave-Redlich-Kwong needs no volume translation to sit within 3.5 % of the real fluid
            assert abs(rho / fluid.density - 1.0) < 0.035, (name, p, T, rho, fluid.density)


@pytest.mark.parametrize(
    "p, T, Y",
    [
        (59.761e6, 111.73, dict(O2=1.0)),
        (59.761e6, 130.0, dict(O2=0.3, CH4=0.5, N2=0.2)),
        (5.0e6, 160.0, dict(O2=1.0)),
        (5.0e7, 300.0, dict(O2=0.4, CH4=0.2, N2=0.15, CO2=0.25)),
    ],
    ids=["dense cold", "dense mixture", "near critical", "four species"],
)
@pytest.mark.parametrize("guessoff", [0.0, 0.01, 10.0])
def test_the_conserved_state_comes_back(dense, p, T, Y, guessoff):
    # out through e and rho, back through the Newton on T started from the cell's last temperature or as far off as a stage moves it
    _, pback, Tback = roundtrip(dense, p, T, composition(dense, **Y), guessoff)
    assert pback == pytest.approx(p, rel=1e-8) and Tback == pytest.approx(T, rel=1e-8)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_random_dense_mixture_comes_back(dense, seed):
    # at these pressures the cubic's residual energy is a good part of cv T, so a broken departure term would not close the round trip
    rng = np.random.default_rng(seed)
    p, T = rng.uniform(1e7, 8e7), rng.uniform(300.0, 600.0)
    Y = rng.random(dense.sim.mixture.ns)
    _, pback, Tback = roundtrip(dense, p, T, Y / Y.sum(), guessoff=10.0)
    assert pback == pytest.approx(p, rel=1e-8) and Tback == pytest.approx(T, rel=1e-8)


@pytest.mark.parametrize("name, p, T", [("N2", 30e6, 200.0), ("O2", 30e6, 200.0)])
def test_the_density_is_the_cubics_own(dense, shifted, name, p, T):
    # no library species carries a translation; the shifted case states 1e-3 m3/kmol on oxygen
    for mb in (dense, shifted):
        sp = mb.sim.mixture.species[name]
        rho, _, _ = roundtrip(mb, p, T, composition(mb, **{name: 1.0}))
        v = cubicvolume(p, T, sp["Tcrit"], sp["pcrit"], sp["acentric"], sp.get("vshift", 0.0))
        assert rho == pytest.approx(sp["MW"] / v, rel=1e-10)
    assert shifted.sim.mixture.species["O2"]["vshift"] == 1e-3


@pytest.mark.parametrize(
    "p, T, Y",
    [(30e6, 200.0, dict(O2=1.0)), (59.761e6, 130.0, dict(O2=0.3, CH4=0.5, N2=0.2))],
    ids=["pure shifted", "shifted mixture"],
)
def test_the_translated_state_comes_back(shifted, p, T, Y):
    # the Newton's volume is the shifted one the state holds plus the translation
    _, pback, Tback = roundtrip(shifted, p, T, composition(shifted, **Y), guessoff=10.0)
    assert pback == pytest.approx(p, rel=1e-8) and Tback == pytest.approx(T, rel=1e-8)
