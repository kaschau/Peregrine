import numpy as np
import pytest

from peregrine.simulator.sem.eddies import EddyField
from peregrine.simulator.sem.march import blob, march

x0, y0, z0 = 0.3, 0.5, 0.2
eps = np.array([[1.0, -1.0, 1.0]])
sigmas = np.array([[[0.2, 0.15, 0.1], [0.25, 0.1, 0.12], [0.18, 0.2, 0.09]]])
# three points under the eddy, then one beyond its reach in y and one in z
ys = np.array([0.5, 0.55, 0.35, 0.9, 0.5])
zs = np.array([0.2, 0.25, 0.2, 0.2, 0.6])
taus = np.linspace(-0.2, 0.6, 41)


def oneEddy(speed):
    return EddyField(
        np.array([[x0, y0, z0]]), eps, sigmas, speeds=np.array([speed]), frames=None
    )


def expected(tau, speed):
    dists = [[x0 - speed * tau, y0 - y, z0 - z] for y, z in zip(ys, zs)]
    return eps[0] * blob(dists, np.repeat(sigmas, ys.shape[0], axis=0))


def test_one_eddy_passes():
    U = 2.0
    raw, diagnostics = march(ys, zs, oneEddy(U), taus, np.ones(ys.shape[0], bool), None)
    # measured identical
    for t, tau in enumerate(taus):
        assert raw[t] == pytest.approx(expected(tau, U), abs=1e-14)
    sx = sigmas[0, :, 0].max()
    passing = (taus > (x0 - sx) / U) & (taus < (x0 + sx) / U)
    assert np.all(raw[~passing] == 0.0)
    assert np.all(raw[passing][:, :3] != 0.0)
    # the two points out of reach see nothing and are reported silent
    assert np.all(raw[:, 3:] == 0.0)
    assert sorted(diagnostics.silent) == [(0.5, 0.6), (0.9, 0.2)]


def test_eddy_at_rest_is_seen_at_every_time():
    raw, _ = march(ys, zs, oneEddy(0.0), taus, np.ones(ys.shape[0], bool), None)
    assert np.all(raw == raw[0])
    assert raw[0] == pytest.approx(expected(0.0, 0.0), abs=1e-14)


def test_blob_is_the_cosine_product():
    # per direction the largest-scale component is cos(pi d / 2 sigma_max), the others ride on it with their own cosine, and the footprint is the product
    rng = np.random.default_rng(3)
    s = rng.uniform(0.2, 1.0, size=(50, 3, 3))
    d = rng.uniform(-0.3, 0.3, size=(50, 3))
    smax, imax = s.max(axis=1), s.argmax(axis=1)
    want = np.ones((50, 3))
    for direction in range(3):
        big = np.cos(np.pi * d[:, direction] / (2.0 * smax[:, direction]))
        for c in range(3):
            own = np.cos(np.pi * d[:, direction] / (2.0 * s[:, c, direction]))
            want[:, c] *= np.where(imax[:, direction] == c, big, big * own)
    assert np.allclose(blob(d, s), want, rtol=1e-14, atol=0.0)
