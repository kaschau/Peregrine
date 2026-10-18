from pathlib import Path

import numpy as np
import pytest

from peregrine.simulator.sem.march import blob

REFERENCES = Path(__file__).parent / "references"


def test_blob_matches_reference():
    # recorded from the numba implementation; local and not regenerable, so without it the test skips
    path = REFERENCES / "blobReference.npz"
    if not path.exists():
        pytest.skip("no local numba reference for blob")
    ref = np.load(path)
    assert np.array_equal(blob(ref["dists"], ref["sigmas"]), ref["fx"])


def test_blob_zero_outside_support():
    rng = np.random.default_rng(7)
    sigmas = rng.uniform(0.2, 1.0, size=(200, 3, 3))
    dists = rng.uniform(-2.0, 2.0, size=(200, 3))
    fx = blob(dists, sigmas)
    outside = np.any(np.abs(dists) > sigmas.max(axis=1), axis=1)
    assert np.all(fx[outside] == 0.0)
    assert np.all(fx[~outside] != 0.0)


def test_blob_is_separable():
    # the directional factors are the values with the other two distances zero
    rng = np.random.default_rng(11)
    sigmas = rng.uniform(0.5, 1.0, size=(300, 3, 3))
    dists = rng.uniform(-0.4, 0.4, size=(300, 3))
    full = blob(dists, sigmas)
    product = np.ones_like(full)
    for direction in range(3):
        only = np.zeros_like(dists)
        only[:, direction] = dists[:, direction]
        product *= blob(only, sigmas)
    assert np.allclose(full, product, rtol=1e-12, atol=0.0)
