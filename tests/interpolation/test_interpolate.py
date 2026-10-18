from types import SimpleNamespace

import numpy as np
import pytest

import peregrine as pg
from peregrine.tools.interpolate import InterpolateTool


def config():
    c = pg.Config()
    c["simulation"]["simulator"] = "euler"
    c["mixture"]["species"] = {"N2": {"MW": 28.0134, "cp0": 1040.0}, "O2": {"MW": 31.9988, "cp0": 918.0}}
    c["mixture"]["eos"] = "cpg"
    c["bcvals"]["wall"] = {"kind": "slp-adia-wall"}
    return c


def field(x):
    X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
    q = np.empty(X.shape + (6,))
    q[..., 0] = 101325.0 * (1.0 + 0.05 * np.sin(2 * np.pi * X) * np.cos(np.pi * Y))
    q[..., 1] = 10.0 * X + 3.0 * Y
    q[..., 2] = 5.0 * Z**2
    q[..., 3] = -2.0 * X * Y
    q[..., 4] = 300.0 + 20.0 * Z
    q[..., 5] = 0.7 + 0.1 * X
    return q


def source():
    return pg.mesher.CubeMesher(mbdims=[2, 1, 1], blkdims=[8, 6, 5], lengths=[1, 1, 1], defaultbc="wall")


# origin and lengths chosen so no center sits midway between two of the source's
def shifted():
    return pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[6, 4, 5], origin=[0.11, 0.13, 0.07], lengths=[0.75, 0.7, 0.77], defaultbc="wall")


@pytest.fixture
def case(tmp_path):
    c = config()
    iovars = pg.simulator.get_simulator(c).iovars.minimal
    ne = len(iovars.privars)
    assert iovars.privars == ["p", "u", "v", "w", "T", "N2"]
    src = pg.multiblock.Restart(ne)
    source().fill(src)
    src.make_metrics()
    for blk in src.blocks:
        blk.q.set(field(blk.xcc.get()))
    pg.writers.GridWriter(src, str(tmp_path / "g.h5"), precision="double").write(src)
    pg.writers.RestartWriter(src, iovars, str(tmp_path / "q.{n:08d}.h5"), "g.h5", "double", cfg=c).write(src, 7, 0.25)
    tgt = pg.multiblock.Grid()
    shifted().fill(tgt)
    pg.writers.GridWriter(tgt, str(tmp_path / "g2.h5"), precision="double").write(tgt)
    pg.write_cfg(c, str(tmp_path / "case.yaml"))
    return SimpleNamespace(path=tmp_path, result=str(tmp_path / "q.00000007.h5"), iovars=iovars, src=src)


def readback(result, iovars):
    reader = pg.readers.RestartReader(result)
    mb = pg.multiblock.Restart(len(iovars.privars))
    pg.readers.GridReader(reader.grid).fill(mb)
    reader.fill(mb, iovars)
    mb.make_metrics()
    return reader, mb


def interpolated(case, grid, function, smooth=0.0):
    reader = pg.readers.RestartReader(case.result)
    src = pg.multiblock.Restart(len(case.iovars.privars))
    pg.readers.GridReader(reader.grid).fill(src)
    reader.fill(src, case.iovars)
    tgt = pg.multiblock.Restart.from_grid(str(case.path / grid), len(case.iovars.privars))
    # the tool's old one flag: nearest, or an rbf by its basis
    options = {} if function == "nearest" else dict(basis=function, smooth=smooth)
    pg.interpolation.get_interpolator("nearest" if function == "nearest" else "rbf", **options).interpolate(src, tgt)
    out = case.path / f"{function}-{grid[:-3]}"
    pg.writers.RestartWriter(tgt, case.iovars, str(out), grid, "double").write(tgt, reader.nrt, reader.tme)
    return readback(f"{out}.h5", case.iovars)


def samples(src):
    pts = np.concatenate([b.xcc.get().reshape(-1, 3) for b in src.blocks])
    vals = np.concatenate([b.q.get().reshape(-1, b.nprims) for b in src.blocks])
    return pts, vals


def test_nearest_onto_the_same_grid_gives_the_values_back(case):
    _, mb = interpolated(case, "g.h5", "nearest")
    for blk in mb.blocks:
        assert np.array_equal(blk.q.get(), case.src.get_blk(blk.nblki).q.get())


def test_nearest_onto_a_shifted_grid_takes_the_nearest_center(case):
    _, mb = interpolated(case, "g2.h5", "nearest")
    pts, vals = samples(case.src)
    (blk,) = mb.blocks
    centers = blk.xcc.get()
    distance = np.linalg.norm(centers.reshape(-1, 1, 3) - pts.reshape(1, -1, 3), axis=-1)
    # the nearest center is the brute-force one, and unambiguous by 3.8e-4
    nearest = np.sort(distance, axis=1)
    assert (nearest[:, 1] - nearest[:, 0]).min() > 1e-4
    expected = vals[distance.argmin(axis=1)].reshape(blk.q.shape)
    assert np.array_equal(blk.q.get(), expected)
    # which is the field within half a cell: 29 % of the way on v = 5 z^2
    error = np.abs(blk.q.get() - field(centers)).max(axis=(0, 1, 2)) / np.abs(field(centers)).max(axis=(0, 1, 2))
    assert error.max() < 0.3


@pytest.mark.parametrize("function,tolerance", [("linear", 1e-14), ("cubic", 1e-13)])
def test_a_radial_basis_reproduces_its_samples(case, function, tolerance):
    # held to the samples exactly (smooth 0): the solve's roundoff, 1e-15 and 1e-14
    _, mb = interpolated(case, "g.h5", function)
    for blk in mb.blocks:
        want = case.src.get_blk(blk.nblki).q.get()
        scale = np.abs(want).max(axis=(0, 1, 2))
        assert (np.abs(blk.q.get() - want).max(axis=(0, 1, 2)) <= tolerance * scale).all()


@pytest.mark.parametrize("function,bound", [("linear", 5e-2), ("cubic", 1e-2)])
def test_a_radial_basis_follows_a_smooth_field_onto_a_shifted_grid(case, function, bound):
    # the worst row against its scale: 2.6e-2 for the linear basis, 5.2e-3 for the cubic
    _, mb = interpolated(case, "g2.h5", function)
    _, vals = samples(case.src)
    (blk,) = mb.blocks
    got, exact = blk.q.get(), field(blk.xcc.get())
    scale = np.abs(exact).max(axis=(0, 1, 2))
    assert (np.abs(got - exact).max(axis=(0, 1, 2)) <= bound * scale).all()
    # a basis may overshoot, so the result is clipped to the samples' range
    assert (got.min(axis=(0, 1, 2)) >= vals.min(axis=0)).all()
    assert (got.max(axis=(0, 1, 2)) <= vals.max(axis=0)).all()


def test_the_tool_carries_the_result_over(case):
    # the values come over to the single precision the tool writes
    out = case.path / "moved"
    args = SimpleNamespace(result=case.result, grid=str(case.path / "g.h5"), out=str(out), cfg=str(case.path / "case.yaml"), kind="nearest", basis=None, smooth=None, deep_search=False)
    InterpolateTool().main(args)
    reader, mb = readback(f"{out}.h5", case.iovars)
    assert reader.nrt == 7 and reader.tme == 0.25
    assert reader.grid == str(case.path / "g.h5")
    for blk in mb.blocks:
        assert np.allclose(blk.q.get(), case.src.get_blk(blk.nblki).q.get(), rtol=6e-8, atol=0)
