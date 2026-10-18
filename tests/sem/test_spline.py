import numpy as np
import pytest

from peregrine.simulator.sem.chunk import eval_spline, spline


def knotsAndValues(n=12, trailing=(4, 3), seed=5):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.uniform(0.5, 1.5, n))
    return x, rng.normal(size=(n,) + trailing)


def piece(c, i, dt):
    c0, c1, c2, c3 = c[0, i], c[1, i], c[2, i], c[3, i]
    return (
        ((c0 * dt + c1) * dt + c2) * dt + c3,
        (3.0 * c0 * dt + 2.0 * c1) * dt + c2,
        6.0 * c0 * dt + 2.0 * c1,
        6.0 * c0,
    )


def test_interpolates_the_knots():
    x, y = knotsAndValues()
    c = spline(x, y)
    assert c.shape == (4, x.shape[0] - 1) + y.shape[1:]
    assert np.array_equal(c[3], y[:-1])
    assert np.allclose(eval_spline(x, c, x), y, rtol=1e-12, atol=1e-12)


def test_smooth_at_interior_knots():
    # value and first two derivatives agree where consecutive pieces meet
    x, y = knotsAndValues()
    c = spline(x, y)
    scale = np.abs(y).max()
    for i in range(x.shape[0] - 2):
        left = piece(c, i, x[i + 1] - x[i])
        right = piece(c, i + 1, 0.0)
        for order in range(3):
            assert np.allclose(left[order], right[order], rtol=1e-9, atol=1e-9 * scale)


def test_not_a_knot_ends():
    # the first two pieces are one cubic, and so are the last two
    x, y = knotsAndValues()
    c = spline(x, y)
    scale = np.abs(c[0]).max()
    assert np.allclose(c[0, 0], c[0, 1], rtol=1e-9, atol=1e-9 * scale)
    assert np.allclose(c[0, -1], c[0, -2], rtol=1e-9, atol=1e-9 * scale)


def test_reproduces_a_cubic():
    x = np.linspace(0.0, 3.0, 9)
    poly = np.array([0.5, -1.0, 2.0, 0.25])
    y = np.polyval(poly, x)
    c = spline(x, y)
    xq = np.linspace(-0.5, 3.5, 101)
    assert np.allclose(
        eval_spline(x, c, xq), np.polyval(poly, xq), rtol=1e-10, atol=1e-10
    )


def test_evaluate_at_many_points_of_many_columns():
    x, y = knotsAndValues(n=8, trailing=(2, 3))
    c = spline(x, y)
    xq = np.array([x[0], 0.5 * (x[2] + x[3]), x[-1]])
    got = eval_spline(x, c, xq)
    assert got.shape == (3, 2, 3)
    assert np.allclose(got[0], y[0]) and np.allclose(got[-1], y[-1])
    dt = xq[1] - x[2]
    assert np.allclose(got[1], piece(c, 2, dt)[0])


def test_needs_four_knots():
    with pytest.raises(ValueError, match="4 knots"):
        spline([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
