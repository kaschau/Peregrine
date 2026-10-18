import CoolProp
import CoolProp.CoolProp as CP
import numpy as np
import pytest

from peregrine.simulator.mixture import Ru

from tests.eos.support import composition, set_tguess, setstate, solver, state

fluids = {"O2": "Oxygen", "N2": "Nitrogen", "CH4": "Methane"}
states = [
    ("O2", 2e6, 120.0),
    ("O2", 6e7, 300.0),
    ("O2", 2e7, 200.0),
    ("O2", 5e6, 400.0),
    ("N2", 6e7, 300.0),
    ("N2", 2e7, 200.0),
    ("N2", 5e6, 400.0),
    ("CH4", 6e7, 300.0),
    ("CH4", 2e7, 200.0),
    ("CH4", 5e6, 400.0),
]


def coolprops():
    species = {}
    for name, fluid in fluids.items():
        s = CoolProp.AbstractState("SRK", fluid)
        species[name] = dict(MW=s.molar_mass() * 1e3, Tcrit=s.T_critical(), pcrit=s.p_critical(), acentric=s.acentric_factor(), vshift=0.0)
    return species


@pytest.fixture(scope="module")
def realgas(backend):
    return solver(backend, coolprops(), "real-gas", (100.0, 1000.0))


@pytest.fixture(scope="module")
def ideal(backend):
    return solver(backend, coolprops(), "tpg", (100.0, 1000.0))


def reference(name, p, T):
    AS = CoolProp.AbstractState("SRK", fluids[name])
    # the phase is imposed since the cubic has three roots at some of these states
    AS.specify_phase(CP.iphase_gas if T > AS.T_critical() else CP.iphase_liquid)
    AS.update(CP.PT_INPUTS, p, T)
    return AS


def agrees(s, ideal, AS, MW, T, p):
    assert s["rho"] == pytest.approx(AS.rhomass(), rel=1e-5)
    # CoolProp's speed of sound sits on its own ideal cp, the refit's differs by 1e-3 at 120 K
    assert s["c"] == pytest.approx(AS.speed_sound(), rel=3e-3)
    assert (s["cp"] - ideal["cp"]) - (AS.cpmass() - AS.cp0mass()) == pytest.approx(0.0, abs=1e-5 * AS.cpmass())
    hres = AS.hmolar_residual() / MW * 1e3
    assert s["h"] - ideal["h"] == pytest.approx(hres, rel=1e-5)
    assert s["e"] - ideal["e"] == pytest.approx(hres - (p / AS.rhomass() - Ru / MW * T), rel=1e-5)


@pytest.mark.parametrize("name, p, T", states)
def test_the_cubic_is_coolprops(realgas, ideal, name, p, T):
    AS = reference(name, p, T)
    Y = composition(realgas, **{name: 1.0})
    MW = realgas.sim.mixture.species[name]["MW"]
    setstate(ideal, p, T, Y)
    si = state(ideal)
    setstate(realgas, p, T, Y)
    agrees(state(realgas), si, AS, MW, T, p)
    # from the conserved state with the Newton started cold at 300 K, measured within 4e-7 in p for the liquid at 120 K
    set_tguess(realgas, to=0.0)
    realgas.postrhs()
    s = state(realgas)
    assert s["p"] == pytest.approx(p, rel=1e-6) and s["T"] == pytest.approx(T, rel=1e-6)
    agrees(s, si, AS, MW, T, p)


def state_at(mb, p, T, Y):
    setstate(mb, p, T, Y)
    return state(mb)


@pytest.mark.parametrize("Y", [dict(O2=1.0), dict(CH4=1.0), dict(N2=1.0), dict(O2=0.3, CH4=0.5, N2=0.2)], ids=["O2", "CH4", "N2", "mixture"])
@pytest.mark.parametrize("p, T", [(60e6, 300.0), (2e6, 120.0), (30e6, 200.0)])
def test_gamma_and_the_sound_speed_are_the_density_derivatives_own(dense, Y, p, T):
    Y = composition(dense, **Y)
    s = state_at(dense, p, T, Y)
    dp, dT = 1e-4 * p, 1e-4 * T
    rho_p = (state_at(dense, p + dp, T, Y)["rho"] - state_at(dense, p - dp, T, Y)["rho"]) / (2 * dp)
    rho_T = (state_at(dense, p, T + dT, Y)["rho"] - state_at(dense, p, T - dT, Y)["rho"]) / (2 * dT)
    gamma = rho_p / (rho_p - (T / s["cp"]) * (rho_T / s["rho"]) ** 2)
    # nitrogen at 120 K and 2 MPa sits by its critical point, where the differences' truncation is the floor (measured 9e-7 there, 8e-8 elsewhere)
    within = 1e-5 if (Y[2] == 1.0 and T == 120.0) else 1e-6
    assert s["gamma"] == pytest.approx(gamma, rel=within)
    assert s["c"] == pytest.approx(np.sqrt(gamma / rho_p), rel=within)
