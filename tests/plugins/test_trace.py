import numpy as np
import pytest

import peregrine as pg

from tests.gases import configure


def box(backend, tmp_path, probe=None, **simulation):
    config = pg.Config()
    configure(config, "air", "navier-stokes")
    # two steps of a fixed size unless the test says otherwise
    config["simulation"].update(dt=1e-6, niter=2)
    config["simulation"].update(simulation)
    config["ics"]["u"] = 10.0
    config["plugins"]["trace"] = {
        "niter-out": 1,
        "traces": {
            "probe": {
                "type": "point",
                "p0": [0.31, 0.52, 0.5],
                "file": str(tmp_path / "p.csv"),
                **(probe or {"niter-out": 2}),
            },
            "line": {
                "type": "line",
                "p0": [0.0, 0.5, 0.5],
                "p1": [1.0, 0.5, 0.5],
                "n": 6,
                "file": str(tmp_path / "line.csv"),
            },
        },
    }
    mesh = pg.mesher.CubeMesher(
        mbdims=[2, 1, 1], blkdims=[6, 6, 6], lengths=[1, 1, 1], periodic=[True] * 3
    )
    return pg.get_solver(config, mesh, backend(config))


def test_points_land_on_their_nearest_cells(backend, tmp_path):
    mb = box(backend, tmp_path)
    plugin = mb.plugins["trace"]
    fname, points = plugin.traces["probe"]
    probes = [(nblki, slot, cell) for nblki, cells in plugin.slots.items() for slot, cell in enumerate(cells) if cell[0] == "probe"]
    ((nblki, slot, (tag, n, i, j, k, kept)),) = probes
    center = mb.get_blk(nblki).xcc.get()[i, j, k]
    # the plugin keeps the cell's center from its start, so an act pulls no centers
    assert np.array_equal(kept, center)
    # the block's list: the header counts its cells, each a code on the allocation's extents, in slot order
    blk = mb.get_blk(nblki)
    n0, n1, _ = blk.tracelist.shape[:3]
    count = len(plugin.slots[nblki])
    head = blk.tracelist.head(2 + count)
    assert head[0] == head[1] == count
    assert head[2 + slot] == i + n0 * (j + n1 * k)
    nearest = min(
        np.linalg.norm(
            b.xcc.get()[b.interior].reshape(-1, 3) - points[0], axis=1
        ).min()
        for b in mb.blocks
    )
    assert np.linalg.norm(center - points[0]) == pytest.approx(nearest)
    # the line's six points span both blocks, every one placed once
    mine = [(n, nblki) for nblki, cells in plugin.slots.items() for tag, n, *_ in cells if tag == "line"]
    assert sorted(n for n, _ in mine) == list(range(6))
    assert {nblki for _, nblki in mine} == {0, 1}


def test_a_trace_appends_a_row_per_point_per_act(backend, tmp_path):
    mb = box(backend, tmp_path)
    mb.run()
    lines = (tmp_path / "line.csv").read_text().splitlines()
    assert lines[0] == "time, x, y, z, " + ", ".join(mb.sim.iovars.privars)
    rows = np.loadtxt(tmp_path / "line.csv", delimiter=",", skiprows=1)
    assert rows.shape == (2 * 6, 4 + mb.sim.ne)
    # the probe names its own cadence: once, after the second step
    probe = np.loadtxt(tmp_path / "p.csv", delimiter=",", skiprows=1, ndmin=2)
    assert probe.shape == (1, 4 + mb.sim.ne) and probe[0, 0] == pytest.approx(mb.tme.get())
    # the last act's rows are the state now, at the cells the points landed on
    last = rows[6:]
    assert np.allclose(last[:, 0], mb.tme.get())
    plugin = mb.plugins["trace"]
    mine = [(n, nblki, i, j, k) for nblki, cells in plugin.slots.items() for tag, n, i, j, k, _ in cells if tag == "line"]
    for n, nblki, i, j, k in mine:
        blk = mb.get_blk(nblki)
        values = mb.sim.iovars.data(blk, mb.sim.iovars.privars)
        assert np.allclose(last[n, 1:4], blk.xcc.get()[i, j, k])
        assert np.allclose(last[n, 4:], [values[v][i, j, k] for v in mb.sim.iovars.privars])


def test_the_cfl_controller_lands_on_dt_out(backend, tmp_path):
    # steps sized by the CFL, none a multiple of the probe's dt-out, are shortened to land on it: every row at a multiple, none twice
    dtout = 3e-4
    simulation = dict(controller="cfl", cfl=0.5, niter=8)
    simulation["dt-max"] = 1e-3
    mb = box(backend, tmp_path, probe={"dt-out": dtout}, **simulation)
    # a uniform flow in a periodic box stays uniform, so the CFL step is one number, here 1.15e-4
    dtcfl = 0.5 / mb.cflmax()["combined"]
    assert 2 * dtcfl < dtout < 3 * dtcfl
    mb.run()
    # what the controller is to do over the eight steps: the CFL step, cut to reach a multiple
    tme, landings = 0.0, []
    for _ in range(8):
        due = (int(tme / dtout + 1e-9) + 1) * dtout
        if due - tme < dtcfl:
            tme, landings = due, landings + [due]
        else:
            tme += dtcfl
    probe = np.loadtxt(tmp_path / "p.csv", delimiter=",", skiprows=1, ndmin=2)
    times = probe[:, 0]
    assert len(times) == len(landings) == 2
    assert np.allclose(times, landings, rtol=1e-12, atol=0.0)
    assert mb.nrt == 8 and mb.tme.get() == pytest.approx(tme, rel=1e-12)
