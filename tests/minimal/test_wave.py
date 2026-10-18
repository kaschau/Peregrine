import numpy as np
import pytest

from tests.minimal.cases import P0, U0, carry_wave

KEPAEC = dict(primary="kepaec")
RUSANOV = dict(primary="rusanov")
KOREN = dict(primary="rusanov", recon="muscl", limiter="koren")

# develop's (max, mean) errors on 40 and 80 cells, which this tree reproduces within 2e-12
DEVELOP = {
    ("kepaec", 41): (0.02575046156384575, 0.016444111866504223),
    ("kepaec", 81): (0.006453447431341797, 0.004112094222018195),
    ("rusanov", 41): (0.6057301945079234, 0.381780696904545),
    ("koren", 41): (0.0483817365005661, 0.015561740308863831),
    ("koren", 81): (0.014509027984775713, 0.0028437132869492147),
}


def wave_err(backend, nx, u=1.0, **keys):
    error, prims, drift = carry_wave(backend, nx, u, **keys)
    # the pressure and velocity uniform and the mass kept on a contact: measured within 7e-14
    assert np.abs(prims[:, 0] / P0 - 1.0).max() < 1e-12
    assert np.abs(prims[:, 1] / (u * U0) - 1.0).max() < 1e-12
    assert abs(drift) < 1e-12
    return error.max(), error.mean()


def test_kepaec_keeps_the_equilibrium_at_second_order(backend):
    e41, e81 = wave_err(backend, 41, **KEPAEC), wave_err(backend, 81, **KEPAEC)
    # second order: measured 3.99 in the max, 4.00 in the mean
    assert e41[0] / e81[0] > 3.5 and e41[1] / e81[1] > 3.5
    assert e41 == pytest.approx(DEVELOP["kepaec", 41], rel=1e-8)
    assert e81 == pytest.approx(DEVELOP["kepaec", 81], rel=1e-8)


def test_koren_on_rusanov_tends_to_third_order(backend):
    e41, e81 = wave_err(backend, 41, **KOREN), wave_err(backend, 81, **KOREN)
    first = wave_err(backend, 41, **RUSANOV)
    # in the mean, away from the clipped crests: measured 2.45; in the max 1.74
    assert np.log2(e41[1] / e81[1]) > 2.1 and np.log2(e41[0] / e81[0]) > 1.5
    # sharper than piecewise constant: measured 0.08 of it
    assert e41[0] < 0.25 * first[0]
    assert e41 == pytest.approx(DEVELOP["koren", 41], rel=1e-8)
    assert e81 == pytest.approx(DEVELOP["koren", 81], rel=1e-8)
    assert first == pytest.approx(DEVELOP["rusanov", 41], rel=1e-8)


@pytest.mark.parametrize("keys", [KOREN, KEPAEC], ids=["koren", "kepaec"])
def test_the_wave_errs_alike_carried_either_way(backend, keys):
    # the limiter biases each side toward its own cell's upwind difference: measured within 2e-13
    forward, backward = wave_err(backend, 41, 1.0, **keys), wave_err(backend, 41, -1.0, **keys)
    assert np.allclose(forward, backward, rtol=1e-10, atol=0.0)


def test_a_switch_that_never_fires_leaves_the_primary(backend):
    # the pressure and velocity are uniform to roundoff, so jameson's sensors stay off: measured within 2e-14
    alone = wave_err(backend, 41, **KEPAEC)
    blended = wave_err(backend, 41, formulation="blended", primary="kepaec", secondary="rusanov", switch="jameson")
    assert np.allclose(blended, alone, rtol=1e-12, atol=0.0)
