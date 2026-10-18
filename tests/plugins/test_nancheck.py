import numpy as np
import pytest

import peregrine as pg

from tests.gases import configure

BAD = [np.nan, np.inf, -np.inf]
DTYPES = {"double": np.float64, "single": np.float32}


def box(backend, precision):
    config = pg.Config()
    configure(config, "air", "euler")
    config["simulation"].update(controller="fixed", dt=1e-6, precision=precision)
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    config["plugins"]["nancheck"] = {}
    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1],
        blkdims=[5, 5, 5],
        lengths=[1e-3] * 3,
        bcnames=dict.fromkeys(range(1, 7), "walls"),
    )
    mb = pg.get_solver(config, mesh, backend(config))
    # the arrays, and so the check's kernel, are in the precision asked for
    assert mb.blocks[0].Q.get().dtype == DTYPES[precision]
    return mb


@pytest.mark.parametrize("precision", DTYPES)
@pytest.mark.parametrize("bad", BAD)
def test_a_non_finite_interior_value_stops_the_run(backend, tmp_path, monkeypatch, precision, bad):
    monkeypatch.chdir(tmp_path)
    mb = box(backend, precision)
    check = mb.plugins["nancheck"]
    # a finite state passes
    check(mb)
    blk = mb.blocks[0]
    ng = blk.ng
    Q = blk.Q.get()
    Q[ng + 1, ng + 2, ng + 1, 4] = bad
    blk.Q.set(Q)
    with pytest.raises(RuntimeError):
        check(mb)
    # the log names the cell, for an infinity as for a nan
    log = (tmp_path / "nans_0.log").read_text().splitlines()
    x, y, z = blk.xcc.get()[ng + 1, ng + 2, ng + 1]
    assert log[1:] == [f"x = {x} y = {y} z = {z}"]


@pytest.mark.parametrize("precision", DTYPES)
@pytest.mark.parametrize("bad", BAD)
def test_a_non_finite_halo_value_is_ignored(backend, tmp_path, monkeypatch, precision, bad):
    monkeypatch.chdir(tmp_path)
    mb = box(backend, precision)
    check = mb.plugins["nancheck"]
    blk = mb.blocks[0]
    ng = blk.ng
    Q = blk.Q.get()
    # the low-i halo layer over the interface proper, which `all` walks and `interior` does not
    Q[0, ng + 2, ng + 2, 4] = bad
    blk.Q.set(Q)
    check(mb)
