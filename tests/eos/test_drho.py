import numpy as np
import pytest

from tests.eos.support import WithEosKernels, cell, setstate, solver, state

# a calorically perfect gas states its constants; the others fit the library's
mixtures = {
    "cpg": {
        "O2": {"MW": 31.998, "cp0": 918.0},
        "N2": {"MW": 28.014, "cp0": 1040.0},
        "CO2": {"MW": 44.009, "cp0": 844.0},
        "CH4": {"MW": 16.043, "cp0": 2220.0},
    },
    "tpg": ["O2", "N2", "CO2", "CH4"],
    "real-gas": ["O2", "N2", "CO2", "CH4"],
    "real-gas shifted": {"O2": {"vshift": 1e-3}, "N2": {}, "CO2": {}, "CH4": {}},
}


def density(mb, p, T, Y):
    setstate(mb, p, T, Y)
    return state(mb)["rho"]


@pytest.mark.parametrize("case", list(mixtures))
def test_derivatives_are_the_states_own(backend, case):
    # a fixed draw: the real-gas composition it gives sits 5% from ideal, which the last check needs
    rng = np.random.default_rng(7)
    eos = case.split()[0]
    mb = solver(backend, mixtures[case], eos, cls=WithEosKernels)
    ns = mb.sim.mixture.ns
    # a dense cold state, where a real gas is far from ideal
    p, T = 60e5, 320.0
    Y = rng.random(ns)
    Y /= Y.sum()
    rho = density(mb, p, T, Y)
    mb.launch(mb.kernels["drho"](Q="Q"), "interior")
    qj = cell(mb, "qj")
    # central differences of the state from primitives
    dp, dT = 1e-3 * p, 1e-3 * T
    rho_p = (density(mb, p + dp, T, Y) - density(mb, p - dp, T, Y)) / (2 * dp)
    rho_T = (density(mb, p, T + dT, Y) - density(mb, p, T - dT, Y)) / (2 * dT)
    assert abs(qj[0] / rho_p - 1) < 1e-5 and abs(qj[1] / rho_T - 1) < 1e-5
    for n in range(ns - 1):
        # Y_n up, the last species down by the same
        d = 1e-3 * min(Y[n], Y[-1])
        up, down = Y.copy(), Y.copy()
        up[n], up[-1] = Y[n] + d, Y[-1] - d
        down[n], down[-1] = Y[n] - d, Y[-1] + d
        rho_Y = (density(mb, p, T, up) - density(mb, p, T, down)) / (2 * d)
        assert abs(qj[2 + n] / rho_Y - 1) < 1e-5, (case, n, qj[2 + n], rho_Y)
    # far from ideal, or the test proved nothing of the cubic (measured 5%)
    if eos == "real-gas":
        assert abs(qj[0] * p / rho - 1) > 0.03
