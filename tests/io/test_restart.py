from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import peregrine as pg
from peregrine.tools import run

from tests.gases import configure, primitives

# every carried species of CH4_O2_FFCMY positive, CO2 taking the rest, so no row of the solution is zero
mixture = {"CH4": 0.2, "O2": 0.6}
mixture.update({s: 0.01 for s in ("H2", "H", "O", "OH", "HO2", "H2O", "CH3", "CO", "CH2O")})


def case(backend, tmp_path, gas="air", integrator="tvd-rk3", writer=None):
    config = pg.Config()
    configure(config, gas, "navier-stokes")
    config["time-integration"]["integrator"] = integrator
    config["simulation"]["dt"] = 1e-8
    config["ics"]["u"] = 10.0
    if gas == "CH4_O2":
        config["ics"]["Y"] = dict(mixture)
    config["bcvals"]["walls"] = {"kind": "no-slp-adia-wall"}
    if writer is not None:
        config["plugins"]["writer"] = dict(writer)
    mesh = pg.mesher.CubeMesher(
        mbdims=[2, 1, 1],
        blkdims=[6, 5, 4],
        lengths=[1, 1, 1],
        periodic=[False, True, True],
        bcnames={1: "walls", 2: "walls"},
    )
    mb = pg.get_solver(config, mesh, backend(config))
    prims = []
    for blk in mb.blocks:
        rng = np.random.default_rng(blk.nblki)
        q = primitives(mb, blk)
        q[..., 0] *= 1 + 0.05 * rng.random(q.shape[:3])
        q[..., 4] *= 1 + 0.1 * rng.random(q.shape[:3])
        prims.append(q)
    mb.set_prims(prims)
    if writer is None:
        pg.writers.GridWriter(mb, str(tmp_path / "g.h5"), precision="double").write(mb)
    return config, mb


def interior(mb):
    return [blk.Q.get()[blk.interior].copy() for blk in mb.blocks]


def scaled_err(got, want):
    scale = np.abs(want).max(axis=(0, 1, 2))
    # the momentum rows share one scale, since a few ulps of energy reach every component through two steps of pressure gradient
    scale[1:4] = scale[1:4].max()
    scale[5:] = scale[0]
    return np.abs(got - want).max(axis=(0, 1, 2)) / scale


def restarted(backend, tmp_path, name):
    reader = pg.readers.RestartReader(str(tmp_path / name))
    return reader, pg.get_solver(reader.cfg, pg.readers.GridReader(reader.grid), backend(reader.cfg), reader)


@pytest.mark.parametrize(
    "gas, integrator, precision, tolerance",
    [
        # the state goes out through the primitives and back through the eos, a few ulps (measured 5.8e-16 and 4.3e-16)
        ("air", "tvd-rk3", "double", 5e-14),
        ("air", "rk4", "double", 5e-14),
        # the thermally perfect gas's temperature is a fixed count of Newton steps, whose residual the round trip drops and the formation enthalpies magnify (measured 3.2e-12)
        ("CH4_O2", "tvd-rk3", "double", 1e-11),
        # a single-precision file: the rounding of every primitive, the species' through the formation enthalpies (measured 1.8e-7 and 8.2e-7)
        ("air", "tvd-rk3", "single", 2e-6),
        ("CH4_O2", "tvd-rk3", "single", 2e-6),
    ],
)
def test_a_restart_continues_as_the_case_would_have(backend, tmp_path, gas, integrator, precision, tolerance):
    config, mb = case(backend, tmp_path, gas, integrator)
    dt = config["simulation"]["dt"]
    for _ in range(3):
        mb.step(dt)
    writer = pg.writers.RestartWriter(mb, mb.sim.iovars, str(tmp_path / "q.{n:08d}.h5"), "g.h5", precision, cfg=config)
    writer.write(mb, mb.nrt, mb.tme.get())
    written, tme = interior(mb), mb.tme.get()
    # the same case, from the file: what the first does next, it does
    for _ in range(2):
        mb.step(dt)
    expect = interior(mb)

    reader, again = restarted(backend, tmp_path, "q.00000003.h5")
    assert reader.privars == mb.sim.iovars.privars and reader.nrt == 3
    assert type(again.integ).name == integrator
    assert again.nrt == 3 and again.tme.get() == tme
    for a, b in zip(interior(again), written):
        assert scaled_err(a, b).max() < tolerance
    for _ in range(2):
        again.step(dt)
    for a, b in zip(interior(again), expect):
        assert scaled_err(a, b).max() < tolerance
    assert again.tme.get() == mb.tme.get()


def test_the_writer_plugins_result_restarts_the_case(backend, tmp_path):
    # the plugin writes the grid beside its results and the result in single precision unless told otherwise
    config, mb = case(backend, tmp_path, writer={"dir": str(tmp_path)})
    dt = config["simulation"]["dt"]
    for _ in range(3):
        mb.step(dt)
    mb.plugins["writer"](mb)
    written, tme = interior(mb), mb.tme.get()
    with h5py.File(tmp_path / "q.00000003.h5") as f:
        assert f["results_000000/p"].dtype == np.float32
        assert f.attrs["grid"] == "g.h5" and (tmp_path / "g.h5").exists()
    reader, again = restarted(backend, tmp_path, "q.00000003.h5")
    assert again.nrt == 3 and again.tme.get() == tme
    # measured 1.8e-7, the single-precision file's rounding
    for a, b in zip(interior(again), written):
        assert scaled_err(a, b).max() < 2e-6


def test_the_file_holds_every_export_variable(backend, tmp_path):
    config, mb = case(backend, tmp_path, gas="CH4_O2")
    writer = pg.writers.RestartWriter(mb, mb.sim.iovars, str(tmp_path / "q.{n:08d}.h5"), "g.h5", "double")
    writer.write(mb, mb.nrt, mb.tme.get())
    with h5py.File(tmp_path / "q.00000000.h5") as f:
        variables = [v.decode() for v in f.attrs["variables"]]
        assert variables == mb.sim.iovars.exported
        assert [v.decode() for v in f.attrs["privars"]] == mb.sim.iovars.privars
        group = f["results_000000"]
        assert set(group) == set(variables)
        blk = mb.blocks[0]
        data = mb.sim.iovars.data(blk, variables)
        for name in variables:
            assert np.array_equal(group[name][...], data[name][blk.interior].T)
        # the last species is what the others leave
        total = sum(group[s][...] for s in mb.sim.mixture.spnames)
        assert np.abs(total - 1.0).max() < 1e-15


def test_the_run_tool_restarts_from_the_result_alone(backend, tmp_path):
    config, mb = case(backend, tmp_path)
    config["simulation"]["niter"] = 2
    writer = pg.writers.RestartWriter(mb, mb.sim.iovars, str(tmp_path / "q.{n:08d}.h5"), "g.h5", "double", cfg=config)
    writer.write(mb, mb.nrt, mb.tme.get())
    # no config and no grid beside the result: both come from it; the backend is named, as on the command line
    run.simulate(SimpleNamespace(restart=str(tmp_path / "q.00000000.h5"), cfg=None, grid=None, backend=backend.name))
