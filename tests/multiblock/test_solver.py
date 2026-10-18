import numpy as np
import pytest

import peregrine as pg

from tests.gases import air, configure, primitives


def config(physics="navier-stokes", gas="air"):
    c = pg.Config()
    configure(c, gas, physics)
    c["bcvals"]["wall"] = {"kind": "slp-adia-wall"}
    return c


def sided(names={}, blkdims=[5, 5, 5]):
    return pg.mesher.CubeMesher(
        mbdims=[1, 1, 1], blkdims=blkdims, lengths=[1, 1, 1], bcnames=names, defaultbc="wall"
    )


def test_named_boundaries_take_the_configs_values(backend):
    c = config()
    c["bcvals"]["inlet"] = {
        "kind": "sub-in-fvt",
        "u": 3.0,
        "v": 0.0,
        "w": 0.0,
        "T": 310.0,
    }
    c["bcvals"]["exit"] = {"kind": "sub-out-fp", "p": 90000.0}
    mb = pg.get_solver(c, sided({1: "inlet", 2: "exit"}), backend(c))
    blk = mb.blocks[0]
    inlet, exit_, wall = (mb.bccolls[name] for name in ("inlet", "exit", "wall"))
    assert inlet.kind.kind == "sub-in-fvt" and inlet.entries == [blk.get_intface(1)]
    assert exit_.kind.kind == "sub-out-fp" and exit_.entries == [blk.get_intface(2)]
    assert wall.entries == [blk.get_intface(n) for n in range(3, 7)]
    # the entry's values are the kind's, written into its kernels as numbers
    options = inlet.kind.options()
    assert options["T"] == "310.0" and options["u"] == "3.0"
    assert exit_.kind.options()["p"] == "90000.0"
    kernels = inlet.kind.kernels("preadvflux", "Q")
    assert [k.axis for k in kernels] == [0, 1, 2]
    assert all(k.options["T"] == "310.0" for k in kernels)
    # a wall reads nothing; a boundary is of no kind on its own
    assert wall.kind.options() is None
    for n in range(3, 7):
        face = blk.get_intface(n)
        assert face.bcname == "wall" and face.intkind is None


def test_a_name_the_config_does_not_know_is_refused(backend):
    c = config()
    with pytest.raises(ValueError, match="mystery"):
        pg.get_solver(c, sided({1: "mystery"}), backend(c))
    c = config()
    c["bcvals"]["noType"] = {"p": 1.0}
    with pytest.raises(KeyError, match="kind"):
        pg.get_solver(c, sided({1: "noType"}), backend(c))
    # a side the mesher did not name cannot be given a value
    c = config()
    with pytest.raises(ValueError, match="did not name"):
        pg.get_solver(c, pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[5, 5, 5]), backend(c))


def test_a_boundary_the_physics_cannot_have_is_refused(backend):
    c = config("euler")
    c["bcvals"]["sticky"] = {"kind": "no-slp-adia-wall"}
    with pytest.raises(KeyError, match="no-slp-adia-wall"):
        pg.get_solver(c, sided({1: "sticky"}), backend(c))


def test_the_volumes_of_a_unit_cube(backend):
    c = config("euler")
    blk = pg.get_solver(c, sided(), backend(c)).blocks[0]
    assert np.array_equal(1.0 / blk.detJinv.get()[blk.interior], np.full((4, 4, 4), 0.25**3))
    # a grid used as a grid holds every metric too, made from its nodes alone
    grid = pg.multiblock.Grid()
    sided().fill(grid)
    grid.make_metrics()
    assert np.array_equal(grid.blocks[0].detJinv.get(), blk.detJinv.get()[blk.interior])


def test_an_axis_one_cell_thick_is_infinitely_long(backend):
    # a block is not marched across such an axis: no step limit binds on it
    c = config("euler")
    blk = pg.get_solver(c, sided(blkdims=[5, 2, 5]), backend(c)).blocks[0]
    lengths = blk.dijk.get()
    assert np.isinf(lengths[..., 1]).all()
    assert np.allclose(lengths[blk.interior][..., (0, 2)], 1 / 4)


def test_the_cfl_speeds_are_the_hand_values(backend):
    # a uniform box of air at 300 K moving at 50 m/s through cells of 0.25
    c = config("euler")
    c["ics"]["u"] = 50.0
    mb = pg.get_solver(c, sided(), backend(c))
    R = 8314.46261815324 / air["Air"]["MW"]
    cp = air["Air"]["cp0"]
    a = np.sqrt(cp / (cp - R) * R * c["ics"]["T"])
    speeds = mb.cflmax()
    # the acoustic speed counts the three directions together, the convective is the normal speed
    assert speeds["acoustic"] == pytest.approx(a * np.sqrt(3) / 0.25, rel=1e-13)
    assert speeds["convective"] == pytest.approx(50.0 / 0.25, rel=1e-13)
    assert speeds["combined"] == pytest.approx(speeds["acoustic"] + speeds["convective"], rel=1e-13)


def test_export_data_derives_from_the_state(backend):
    c = config(gas="CH4_O2")
    mb = pg.get_solver(c, sided(), backend(c))
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    rng = np.random.default_rng(1)
    q[..., 1:4] = 30 * rng.random(q.shape[:3] + (3,))
    mb.set_prims([q])
    names = mb.sim.iovars.exported
    data = mb.sim.iovars.data(blk, names)
    assert list(data) == names and names[0] == "rho"
    Q = blk.Q.get()
    i = blk.interior
    assert np.array_equal(data["rho"][i], Q[i][..., 0])
    assert np.allclose(data["u"][i], Q[i][..., 1] / Q[i][..., 0], rtol=1e-14)
    species = mb.sim.mixture.spnames
    assert np.allclose(sum(data[s][i] for s in species), 1.0, rtol=1e-14)
    # the thermally perfect gas's temperature is a Newton solve: measured 7e-14
    back = primitives(mb, blk)[i]
    scale = np.abs(q[i]).max(axis=(0, 1, 2))
    assert (np.abs(back - q[i]).max(axis=(0, 1, 2)) <= 1e-13 * scale).all()
