import h5py
import numpy as np
import pytest

import peregrine as pg


def cfg():
    c = pg.Config()
    c["simulation"]["simulator"] = "euler"
    c["simulation"]["dt"] = 1e-7
    c["mixture"]["species"] = {"N2": {"MW": 28.0134, "cp0": 1040.0}, "O2": {"MW": 31.9988, "cp0": 918.0}}
    c["mixture"]["eos"] = "cpg"
    c["ics"]["Y"] = {"N2": 0.77}
    c["ics"]["u"] = 10.0
    c["bcvals"]["wall"] = {"kind": "slp-adia-wall"}
    return c


def mesh():
    return pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[5, 4, 3], lengths=[1, 1, 1], bcnames={n: "wall" for n in range(1, 7)})


def written(backend, tmp_path):
    c = cfg()
    mb = pg.get_solver(c, mesh(), backend(c))
    mb.step(c["simulation"]["dt"])
    pg.writers.GridWriter(mb, str(tmp_path / "g.h5"), precision="double").write(mb)
    pg.writers.RestartWriter(mb, mb.sim.iovars, str(tmp_path / "q.{n:08d}.h5"), "g.h5", "double", cfg=c).write(mb, mb.nrt, mb.tme.get())
    return mb, tmp_path / "q.00000001.h5"


def interior(mb, name):
    blk = mb.blocks[0]
    return mb.sim.iovars.data(blk, [name])[name][blk.interior].copy()


def test_the_map_is_the_physics_and_the_file_follows_it(backend, tmp_path):
    mb, result = written(backend, tmp_path)
    variables = mb.sim.iovars
    assert variables.privars == ["p", "u", "v", "w", "T", "N2"] and mb.nprims == mb.sim.ne == 6
    assert variables.primitives["p"] is None and variables.primitives["N2"] == 0.0
    assert variables.exported == ["rho", "p", "u", "v", "w", "T", "N2", "O2"]
    assert variables.minimal.exported == variables.privars and variables.minimal.derive is None
    assert variables.vectors == {"Velocity": ("u", "v", "w")}
    with h5py.File(result) as f:
        assert [v.decode() for v in f.attrs["variables"]] == variables.exported
        assert sorted(f["results_000000"]) == sorted(variables.exported)
    assert "Velocity" in (tmp_path / "q.00000001.xmf").read_text()


def test_a_result_reads_back_by_name(backend, tmp_path):
    mb, result = written(backend, tmp_path)
    reader = pg.readers.RestartReader(str(result))
    again = pg.get_solver(reader.cfg, pg.readers.GridReader(reader.grid), backend(reader.cfg), reader)
    assert again.nrt == 1 and again.tme.get() == pytest.approx(mb.tme.get())
    for name in mb.sim.iovars.privars:
        assert np.allclose(interior(again, name), interior(mb, name), rtol=1e-12, atol=0.0)


def test_a_missing_species_starts_at_its_default(backend, tmp_path):
    mb, result = written(backend, tmp_path)
    with h5py.File(result, "a") as f:
        del f["results_000000/N2"]
    reader = pg.readers.RestartReader(str(result))
    again = pg.get_solver(reader.cfg, pg.readers.GridReader(reader.grid), backend(reader.cfg), reader)
    assert np.all(interior(again, "N2") == 0.0) and np.all(interior(again, "O2") == 1.0)


def test_a_missing_pressure_is_refused(backend, tmp_path):
    mb, result = written(backend, tmp_path)
    with h5py.File(result, "a") as f:
        del f["results_000000/p"]
    reader = pg.readers.RestartReader(str(result))
    with pytest.raises(ValueError, match="holds no p"):
        pg.get_solver(reader.cfg, pg.readers.GridReader(reader.grid), backend(reader.cfg), reader)
