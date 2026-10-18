import numpy as np
import pytest

from tests.riemann import RiemannProblem

# Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics, Table 4.3: p*, u*, rhoL*, rhoR*
table = {
    0: (0.30313, 0.92745, 0.42632, 0.26557),
    2: (0.00189, 0.00000, 0.02185, 0.02185),
    3: (460.894, 19.5975, 0.57506, 5.99924),
    4: (1691.64, 8.68975, 14.2823, 31.0426),
    5: (460.894, 0.00000, 0.57506, 5.99924),
}


def star(test):
    pM, uM = test.star()
    _, _, rhoL, _ = test.sample(pM, uM, uM - 1e-9)
    _, _, rhoR, _ = test.sample(pM, uM, uM + 1e-9)
    return pM, uM, rhoL, rhoR


@pytest.mark.parametrize("case", sorted(table))
def test_the_star_state_is_toros(case):
    got = star(RiemannProblem.toro(case))
    pM, uM, rhoL, rhoR = table[case]
    # the table's p* of the near-vacuum case has three digits; the solver gives 0.0018939
    assert got[0] == pytest.approx(pM, rel=3e-3 if case == 2 else 1e-4)
    # u* = 0 by symmetry in cases 2 and 5, which the iteration reaches to its tolerance
    assert got[1] == pytest.approx(uM, rel=1e-4, abs=1e-4)
    assert got[2] == pytest.approx(rhoL, rel=1e-4)
    assert got[3] == pytest.approx(rhoR, rel=1e-4)


def ustar_from_side(test, pM, side):
    g = test.gamma
    if side == "L":
        p, rho, u, c, sign = test.pL, test.rhoL, test.uL, test.cL, -1.0
    else:
        p, rho, u, c, sign = test.pR, test.rhoR, test.uR, test.cR, 1.0
    if pM < p:
        f = 2.0 * c / (g - 1.0) * ((pM / p) ** ((g - 1.0) / (2.0 * g)) - 1.0)
    else:
        A, B = 2.0 / ((g + 1.0) * rho), (g - 1.0) / (g + 1.0) * p
        f = (pM - p) * np.sqrt(A / (pM + B))
    return u + sign * f


@pytest.mark.parametrize("case", range(6))
def test_the_star_state_joins_both_waves(case):
    # every case, the one not in the table too: the two sides' wave relations meet at u*
    test = RiemannProblem.toro(case)
    pM, uM = test.star()
    assert ustar_from_side(test, pM, "L") == pytest.approx(uM, abs=1e-6 * max(1.0, abs(uM)))
    assert ustar_from_side(test, pM, "R") == pytest.approx(uM, abs=1e-6 * max(1.0, abs(uM)))


def test_sods_waves_are_where_the_star_state_puts_them():
    test = RiemannProblem.toro(0)
    g = test.gamma
    x = np.linspace(0.0, 1.0, 2001)
    sol = test.solve(x)
    pM, uM = test.star()
    contact = test.x0 + uM * test.t
    g1, g2 = (g - 1.0) / (2.0 * g), (g + 1.0) / (2.0 * g)
    shock = test.x0 + (test.uR + test.cR * np.sqrt(g2 * pM / test.pR + g1)) * test.t
    # the only two density jumps above 0.02 per point are the contact and the shock
    jumps = x[np.where(np.abs(np.diff(sol["rho"])) > 0.02)[0]]
    assert len(jumps) == 2
    assert jumps[0] == pytest.approx(contact, abs=1e-3) and jumps[1] == pytest.approx(shock, abs=1e-3)
    assert abs(contact - 0.68549) < 1e-5 and abs(shock - 0.85043) < 1e-5
    # the rarefaction: undisturbed ahead of its head, isentropic through it to the contact
    head = test.x0 + (test.uL - test.cL) * test.t
    tail = test.x0 + (uM - test.cL * (pM / test.pL) ** g1) * test.t
    assert (sol["rho"][x < head] == test.rhoL).all()
    entropy = sol["p"] / sol["rho"] ** g
    assert np.allclose(entropy[x < contact - 1e-3], test.pL / test.rhoL**g, rtol=1e-12, atol=0.0)
    between = (x > tail + 1e-3) & (x < contact - 1e-3)
    assert np.allclose(sol["rho"][between], 0.42632, rtol=1e-4, atol=0.0)
    assert np.allclose(sol["u"][between], uM, rtol=1e-12, atol=0.0)
    # the density changes through the fan, so it was sampled, not skipped
    fan = (x > head + 1e-3) & (x < tail - 1e-3)
    assert sol["rho"][fan].max() < test.rhoL and sol["rho"][fan].min() > 0.42632
