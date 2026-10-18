import CoolProp.CoolProp as CP
import numpy as np
import pytest

from tests.eos.support import composition, cubicroots, cubic_ab, set_tguess, leastgibbs, setstate, state, textbookdensity

fluids = {"O2": "Oxygen", "CH4": "Methane"}
# (species, p, T, phase): O2's saturation pressure on SRK is 1.03 MPa at 120 K and 2.82 MPa at 140 K
states = [
    ("O2", 1e6, 140.0, "vapor"),
    ("O2", 2.6e6, 140.0, "vapor"),
    ("O2", 2e6, 120.0, "liquid"),
    ("O2", 3e6, 140.0, "liquid"),
    ("CH4", 0.5e6, 150.0, "vapor"),
    ("CH4", 0.3e6, 140.0, "vapor"),
    ("CH4", 1.5e6, 150.0, "liquid"),
    ("CH4", 1.0e6, 140.0, "liquid"),
]


@pytest.mark.parametrize("name, p, T, phase", states)
def test_the_root_is_the_stable_phases(dense, name, p, T, phase):
    sp = [dense.sim.mixture.species[name]]
    Y = np.array([1.0])
    A, B, _ = cubic_ab(sp, p, T, Y)
    roots = cubicroots(A, B)
    # three roots above the covolume, or the Gibbs rule decided nothing
    assert np.sum(roots > B) == 3
    Z = leastgibbs(roots, A, B)
    assert (Z == roots.max()) == (phase == "vapor")
    setstate(dense, p, T, composition(dense, **{name: 1.0}))
    rho = state(dense)["rho"]
    assert rho == pytest.approx(textbookdensity(sp, p, T, Y), rel=1e-10)
    # measured 1e-4 for oxygen and 7e-3 for methane, whose library critical point (190.7 K, 4.63 MPa) is not CoolProp's (190.564 K, 4.5992 MPa)
    within = 2e-4 if name == "O2" else 1e-2
    assert rho == pytest.approx(CP.PropsSI("Dmass", "T", T, "P", p, f"SRK::{fluids[name]}"), rel=within)
    # back from the solution with the Newton ten kelvin off: the pressure comes from the volume with no root to pick, so the density is the same to the bit
    set_tguess(dense, by=10.0)
    dense.postrhs()
    back = state(dense)
    assert back["p"] == pytest.approx(p, rel=1e-8) and back["T"] == pytest.approx(T, rel=1e-8)
    assert back["rho"] == rho
