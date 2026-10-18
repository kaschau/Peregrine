import numpy as np
import pytest

from tests.simulator import support as sp
from tests.simulator.support import sod  # noqa: F401

# the tree's compositions, and the correction's, kepaec with the matrix dissipation added by Jameson's sensors at the blend's gains
schemes = {
    **sp.schemes,
    "kepaec+matrix-dissipation by jameson": dict(
        formulation="correction",
        primary="kepaec",
        recon="constant",
        correction="matrix-dissipation",
        switch="jameson",
        values={"p-gain": 5.0, "gamma-gain": 1.0},
    ),
}


def wave(backend, name, nx, **kw):
    return sp.wave_errors(backend, nx, **schemes[name], **kw)


###############################################################################
# The density wave, once around a periodic line of 40 and 80 cells
###############################################################################

# measured (max order, mean order, e80): kepaec 2.00, 2.00, 3.872e-3; kepaec4 3.99, 4.00, 4.774e-6; muscl-koren-rusanov 1.81, 2.60, 1.298e-2; muscl-koren-ausmup 1.63, 2.20, 6.922e-3; rusanov 0.39, 0.41, 4.069e-1; ausmup 0.83, 0.83, 1.312e-1; hllc the same as ausmup, both the upwind flux on a contact, and so muscl-koren-hllc 1.63, 2.20, 6.922e-3
orders = [
    ("kepaec", 1.95, 1.95, 3.872e-3),
    ("kepaec4", 3.9, 3.9, 4.774e-6),
    ("muscl-koren-rusanov", 1.6, 2.0, 1.298e-2),
    ("muscl-koren-ausmup", 1.5, 2.0, 6.922e-3),
    ("muscl-koren-hllc", 1.5, 2.0, 6.922e-3),
    ("rusanov", 0.3, 0.3, 4.069e-1),
    ("ausmup", 0.7, 0.7, 1.312e-1),
    ("hllc", 0.7, 0.7, 1.312e-1),
]


@pytest.mark.parametrize("name,atleast,mean_atleast,e80", orders, ids=[o[0] for o in orders])
def test_the_order_on_the_wave(backend, name, atleast, mean_atleast, e80):
    # a limiter clips the crests, so the max norm is under two, and the mean norm away from them is where the kappa = 1/3 scheme tends toward third order
    e40, e80got = wave(backend, name, 41), wave(backend, name, 81)
    assert np.log2(e40[0] / e80got[0]) >= atleast
    assert np.log2(e40[1] / e80got[1]) >= mean_atleast
    assert e80got[0] <= 1.2 * e80


@pytest.mark.parametrize("riemann", ["rusanov", "ausmup", "hllc"])
def test_muscl_is_sharper_than_piecewise_constant(backend, riemann):
    # measured a twelfth, an eleventh and an eleventh
    assert wave(backend, f"muscl-koren-{riemann}", 41)[0] < 0.25 * wave(backend, riemann, 41)[0]


@pytest.mark.parametrize("name", ["kepaec", "rusanov", "ausmup", "hllc", "muscl-koren-rusanov", "muscl-koren-ausmup", "muscl-koren-hllc"])
def test_the_wave_errs_alike_leftward_and_rightward(backend, name):
    # an upwind-biased limiter biases each side toward its own cell's upwind difference, so the wave advected leftward errs as it does rightward; measured within 5e-13
    assert np.allclose(wave(backend, name, 41, u=-100.0), wave(backend, name, 41, u=100.0), rtol=1e-11, atol=0.0)


def test_a_near_zero_weight_leaves_the_primary(backend):
    # the wave's pressure and velocity are uniform to rounding and its gamma one gas's, so neither sensor fires beyond rounding and the blend is the primary's flux; measured within 4e-13
    alone, blended = wave(backend, "kepaec", 41), wave(backend, "kepaec+rusanov by jameson", 41)
    assert np.allclose(blended, alone, rtol=1e-11, atol=0.0)


###############################################################################
# The halo follows the cells the pieces declare
###############################################################################
@pytest.mark.parametrize(
    "name,ng",
    [("kepaec", 1), ("rusanov", 1), ("muscl-koren-rusanov", 2), ("kepaec+rusanov by jameson", 2), ("kepaec+matrix-dissipation by jameson", 2)],
)
def test_the_halo_is_as_deep_as_the_stencil(backend, name, ng):
    # one cell either side for a flux function of the two cells, two for muscl's slopes, and two for Jameson's second difference of the pressure whatever the reconstruction
    assert sp.line(backend, 11, **schemes[name]).ng == ng


###############################################################################
# Toro's shock tubes on a walled line, against the exact solution
###############################################################################

# per test the length, the cells per unit length and each composition's measured mean density error over the fan, locked at a quarter above; test 5 streams from the right wall at Mach 166 and leaves a vacuum there, which the first-order fluxes ride out and the limited sides do not, so no MUSCL pair runs on it
tubes = {
    0: dict(length=1.0, cells=200, rho={"rusanov": 5.242e-2, "ausmup": 4.704e-2, "hllc": 3.822e-2, "muscl-koren-rusanov": 9.576e-3, "muscl-koren-ausmup": 8.479e-3, "muscl-koren-hllc": 7.295e-3, "kepaec+rusanov by jameson": 1.814e-2, "kepaec+matrix-dissipation by jameson": 2.111e-2}),
    1: dict(length=2.0, cells=200, rho={"rusanov": 6.451e-2, "hllc": 3.410e-2, "muscl-koren-rusanov": 1.280e-2, "muscl-koren-ausmup": 8.705e-3, "muscl-koren-hllc": 8.921e-3}),
    2: dict(length=2.5, cells=200, rho={"rusanov": 6.984e-2, "hllc": 7.255e-2, "muscl-koren-rusanov": 7.062e-3, "muscl-koren-ausmup": 1.237e-2, "muscl-koren-hllc": 7.066e-3}),
    3: dict(length=1.0, cells=400, rho={"rusanov": 1.935e-1, "ausmup": 1.611e-1, "hllc": 1.606e-1, "muscl-koren-rusanov": 7.468e-2, "kepaec+rusanov by jameson": 8.125e-2, "kepaec+matrix-dissipation by jameson": 1.113e-1, "muscl-koren-ausmup": 8.666e-2, "muscl-koren-hllc": 8.740e-2}),
    4: dict(length=3.0, cells=200, rho={"rusanov": 1.238e-1, "hllc": 9.191e-2, "muscl-koren-rusanov": 3.490e-2, "muscl-koren-ausmup": 2.667e-2, "muscl-koren-hllc": 2.980e-2}),
    5: dict(length=3.5, cells=200, rho={"rusanov": 2.116e-1, "hllc": 6.203e-2}),
}
# the plateau's p and u against the star state by reconstruction: Sod's and test 1's plateaus are wide, so muscl pins the star state to a part in five hundred (measured 6.7e-4 at worst) and the first-order solvers to one in a hundred (4.0e-3, ausmup on Sod); the strong tests' plateaus are a few cells smeared into by every solver (1.0e-2 at worst); test 2's near-vacuum plateau is not resolved at this spacing and only u* = 0 is asked of it
plateau = {0: dict(constant=1e-2, muscl=2e-3), 1: dict(constant=5e-3, muscl=2e-3), 3: dict(constant=2e-2, muscl=2e-2), 4: dict(constant=2e-2, muscl=2e-2), 5: dict(constant=2e-2, muscl=2e-2)}


def runs():
    for case_number, tube in tubes.items():
        for name in tube["rho"]:
            yield pytest.param(case_number, name, id=f"toro{case_number}-{name}")


@pytest.mark.parametrize("case_number,name", list(runs()))
def test_the_tube_against_the_exact_solution(backend, case_number, name):
    tube = tubes[case_number]
    out = sp.toro(backend, case_number, length=tube["length"], cells_per_unit=tube["cells"], **schemes[name])
    assert out.finite
    # the walls pass nothing: the interior mass and energy are kept (measured 1.6e-13 at worst)
    assert out.mass < 1e-12 and out.energy < 1e-12
    assert out.rho_err <= 1.25 * tube["rho"][name]
    if case_number == 2:
        # the near-vacuum plateau: u* = 0, within a twentieth of the states' speed (measured 3.2e-2)
        assert out.u_err < 5e-2
        return
    if name == "kepaec+rusanov by jameson":
        # the central scheme's wiggles stay where the switch is off: within 6e-2 on Sod (measured 3.9e-2)
        within = 6e-2 if case_number == 0 else None
    elif name == "kepaec+matrix-dissipation by jameson":
        # the correction damps the wiggles at the sensor's level and leaves the rest: within 8e-2 on Sod (measured 6.3e-2)
        within = 8e-2 if case_number == 0 else None
    else:
        within = plateau[case_number]["muscl" if "muscl" in name else "constant"]
    if within is not None:
        assert out.p_err < within and out.u_err < within


def test_the_switch_captures_the_shock(backend):
    # the central scheme alone rings across Sod's tube about as far from the exact solution as rusanov's smearing; blended by the switch it is sharper than either
    alone, blended = sod(backend, **schemes["kepaec"]), sod(backend, **schemes["kepaec+rusanov by jameson"])
    assert blended.rho_err < 0.6 * alone.rho_err
    assert blended.rho_err < 0.6 * sod(backend, **schemes["rusanov"]).rho_err
