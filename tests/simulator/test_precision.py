import ctypes

import numpy as np
import pytest

from tests.simulator import support as sp


def interior(mb):
    return [blk.Q.get()[sp.interior(blk)].astype(np.float64) for blk in mb.blocks]


def rowgap(double, single):
    gaps = [np.abs(d - s).max(axis=(0, 1, 2)) / np.abs(d).max(axis=(0, 1, 2)) for d, s in zip(double, single)]
    return np.max(gaps, axis=0)


def test_the_case_is_single_end_to_end(backend):
    double, single = sp.gatebox(backend, "air", "double"), sp.gatebox(backend, "air", "single")
    assert double.blocks[0].Q.dtype == np.float64 and single.blocks[0].Q.dtype == np.float32
    assert single.backend.fpctype is ctypes.c_float and single.tplargs["fpdtype"] == "float"
    # every kernel a graph launches, and the one reduction the controller reads
    kernels = [k for g in single.graphs.values() for k in g.kernels]
    assert kernels
    assert all("using fpdtype = float;" in k.text for k in kernels)
    speeds = single.cflmax()
    assert set(speeds) == {"acoustic", "convective", "combined"}
    assert all(type(v) is np.float32 and np.isfinite(v) and v > 0.0 for v in speeds.values())
    # measured within 2e-7 of the double case's
    reference = double.cflmax()
    assert all(abs(speeds[k] / reference[k] - 1.0) < 1e-5 for k in reference)


# per gas the gap right after the state is set and after twelve steps of 5e-7 s, when the state has moved by a few percent; measured over the rows: the calorically perfect gas at float32's 2e-7 and 1.3e-6, the thermally perfect gas's energy row at 9.4e-5 from the start, its enthalpy polynomial in ln T evaluated in single, and its momentum rows following through the pressure to 7.8e-4
within = {"air": (1e-6, 5e-6), "CH4_O2": (2e-4, 2e-3)}


@pytest.mark.parametrize("gas", sorted(within))
def test_single_follows_double(backend, gas):
    dt = 5e-7
    double, single = sp.gatebox(backend, gas, "double", dt), sp.gatebox(backend, gas, "single", dt)
    at_start, after = within[gas]
    gap = rowgap(interior(double), interior(single))
    assert gap.max() < at_start
    # every row but the energy is at float32's own precision (measured 3e-7)
    assert np.delete(gap, 4).max() < 1e-6
    for _ in range(12):
        double.step(dt)
        single.step(dt)
    moved = interior(double)
    assert all(np.isfinite(s).all() for s in interior(single))
    assert rowgap(moved, interior(single)).max() < after
