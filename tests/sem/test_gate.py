import numpy as np
import pytest

from tests.sem.gate import CASES, REFERENCES, run

SIGNAL_RTOL = 1e-10
SIGNAL_ATOL_SCALE = 1e-13


@pytest.mark.parametrize("case", sorted(CASES))
def test_gate(case):
    path = REFERENCES / f"{case}.npz"
    if not path.exists():
        pytest.skip(f"no local reference; run: python -m tests.sem.gate --write {case}")
    ref = np.load(path)
    got = run(case)

    assert str(got["params"]) == str(ref["params"])
    for key in (
        "ys",
        "zs",
        "neddy",
        "eddyLocsHead",
        "epsHead",
        "sigmasHead",
        "eddySum",
    ):
        assert np.array_equal(got[key], ref[key]), f"{key} differs"

    for key in ("raw", "up", "vp", "wp"):
        scale = np.max(np.abs(ref[key]))
        assert np.allclose(
            got[key], ref[key], rtol=SIGNAL_RTOL, atol=SIGNAL_ATOL_SCALE * scale
        ), f"{key} differs; max abs diff {np.max(np.abs(got[key] - ref[key]))}"
