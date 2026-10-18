import numpy as np
import pytest

import peregrine as pg
from peregrine.launch.ranges import CellCenterRange

# the reconstructions, with the halo depth each reads
recons = {"constant": 1, "muscl": 2}


def configured(recon="constant", simulator="euler"):
    config = pg.Config()
    config["simulation"]["simulator"] = simulator
    config["mixture"]["species"] = {
        "N2": {"MW": 28.0134, "cp0": 1040.0, "mu0": 1.8e-5, "kappa0": 0.026},
        "O2": {"MW": 31.9988, "cp0": 918.0, "mu0": 2.0e-5, "kappa0": 0.027},
    }
    config["mixture"]["eos"] = "cpg"
    if simulator == "navier-stokes":
        config["mixture"]["trans"] = "constant"
    config["ics"]["Y"] = {"N2": 0.77}
    config["bcvals"]["wall"] = {"kind": "slp-adia-wall"}
    config["time-integration"]["integrator"] = "tvd-rk3"
    if recon == "muscl":
        config["adv-flux"].update(recon="muscl", limiter="koren")
    return config


# an exchanged array trading one plane, as a gradient does, without the physics that declares one
class WithPlaneProbe(pg.multiblock.Solver):
    def _decl_arrays(self):
        super()._decl_arrays()
        self.decl_array("probe", CellCenterRange, comps=2, exchanged=1)


def pair(backend, recon, periodic=(False, False, False), cls=pg.multiblock.Solver):
    mesh = pg.mesher.CubeMesher(mbdims=[2, 1, 1], blkdims=[5, 4, 3], lengths=[2, 1, 1], periodic=list(periodic), defaultbc="wall")
    config = configured(recon)
    mb = pg.get_solver(config, mesh, backend(config), cls=cls)
    assert mb.ng == recons[recon]
    return mb


def stamp(mb, name):
    for blk in mb.blocks:
        a = getattr(blk, name).get()
        i, j, k = np.indices(a.shape[:3])
        cell = 1000 * blk.nblki + 100 * i + 10 * j + k
        a[...] = cell.reshape(cell.shape + (1,) * (a.ndim - 3))
        getattr(blk, name).set(a)


def exchange(mb, name):
    g = mb.backend.graph("test")
    g.add_send("sent", name)
    g.add_recv("landed", name)
    g.bind(mb)
    g.run()


@pytest.fixture(params=list(recons))
def recon(request):
    return request.param


def test_the_halo_holds_the_neighbors_cells(backend, recon):
    mb = pair(backend, recon)
    stamp(mb, "Q")
    exchange(mb, "Q")
    a, b = mb.blocks
    ng = mb.ng
    Qa, Qb = a.Q.get(), b.Q.get()
    # block 0's high i halo is block 1's first interior planes, layer 0 nearest the face, and the other way
    for g in range(ng):
        assert np.array_equal(Qa[-ng + g, ng:-ng, ng:-ng], Qb[ng + g, ng:-ng, ng:-ng])
        assert np.array_equal(Qb[g, ng:-ng, ng:-ng], Qa[-2 * ng + g, ng:-ng, ng:-ng])
    # a wall's halo is not the exchange's to touch: still its own stamp
    j, k = np.indices(Qa.shape[1:3])
    for g in range(ng):
        assert np.array_equal(Qa[g, ..., 0], 100 * g + 10 * j + k)


def test_every_bank_goes_through_the_solutions_exchange(backend, recon):
    mb = pair(backend, recon)
    ng = mb.ng
    assert mb.regs == ("Q", "Q1", "Q2")
    # one message per family, every bank's halos travel by the solution's
    assert sorted(mb.msgs) == ["Q", "nodes"]
    for bank in mb.regs[1:]:
        assert mb.columns[bank].name == "Q"
        stamp(mb, bank)
        exchange(mb, bank)
        a, b = (getattr(blk, bank).get() for blk in mb.blocks)
        for g in range(ng):
            assert np.array_equal(a[-ng + g, ng:-ng, ng:-ng], b[ng + g, ng:-ng, ng:-ng])
            assert np.array_equal(b[g, ng:-ng, ng:-ng], a[-2 * ng + g, ng:-ng, ng:-ng])


def test_an_array_trades_as_many_planes_as_it_says(backend, recon):
    mb = pair(backend, recon, cls=WithPlaneProbe)
    c = mb.interiorcoll
    i = c.entries[0]
    assert c.planes(i, mb.arrays["probe"])[0] == 1 and c.planes(i, mb.arrays["Q"])[0] == mb.ng
    stamp(mb, "probe")
    before = [blk.probe.get().copy() for blk in mb.blocks]
    exchange(mb, "probe")
    after = [blk.probe.get() for blk in mb.blocks]
    ng = mb.ng
    # the layer next to the face came across on both sides of the seam
    assert np.array_equal(after[0][-ng, ng:-ng, ng:-ng], after[1][ng, ng:-ng, ng:-ng])
    assert np.array_equal(after[1][ng - 1, ng:-ng, ng:-ng], after[0][-ng - 1, ng:-ng, ng:-ng])
    # the deeper layers did not; at one layer there are none
    if ng > 1:
        assert np.array_equal(after[0][-1], before[0][-1])
        assert np.array_equal(after[1][0], before[1][0])
    assert np.array_equal(after[0][-ng - 1 :: -1], before[0][-ng - 1 :: -1])


def test_the_local_pair_has_no_buffer(backend, recon):
    mb = pair(backend, recon)
    a, b = mb.blocks
    left, right = a.get_intface(2), b.get_intface(1)
    assert mb.nbrintface(left) is right and mb.nbrintface(right) is left
    c = mb.interiorcoll
    assert c.entries == [left, right]
    assert c.bufs == {} and all(c.nbrs[i] is not None for i in c.entries)
    assert mb.mpicolls == [] and mb.msgs["Q"].ranks == []


def test_periodic_halos_wrap_around(backend, recon):
    mb = pair(backend, recon, periodic=(True, False, False))
    stamp(mb, "Q")
    exchange(mb, "Q")
    a, b = mb.blocks
    ng = mb.ng
    # block 0's low i halo is block 1's last interior planes, and the other way round
    for g in range(ng):
        assert np.array_equal(a.Q.get()[ng - 1 - g, ng:-ng, ng:-ng], b.Q.get()[-ng - 1 - g, ng:-ng, ng:-ng])
        assert np.array_equal(b.Q.get()[-ng + g, ng:-ng, ng:-ng], a.Q.get()[ng + g, ng:-ng, ng:-ng])


def annulus(backend, recon):
    mesher = pg.mesher.AnnulusMesher(mbdims=[1, 1, 2], blkdims=[4, 4, 4], p1=[0, 0, 0], p2=[1, 0, 0], p3=[0, 0.5, 0], sweep=90, thickness=0.5, periodic=True, defaultbc="wall")
    config = configured(recon)
    return pg.get_solver(config, mesher, backend(config))


def radial(x):
    y, z = x[..., 1], x[..., 2]
    return np.stack([np.zeros_like(y), y, z], axis=-1) / np.hypot(y, z)[..., None]


def azimuthal(x):
    y, z = x[..., 1], x[..., 2]
    return np.stack([np.zeros_like(y), -z, y], axis=-1) / np.hypot(y, z)[..., None]


def test_a_rotational_periodic_rotates_the_vectors(backend, recon):
    mb = annulus(backend, recon)
    ng = mb.ng
    a, b = mb.blocks
    for blk in mb.blocks:
        Q = blk.Q.get()
        Q[..., 0], Q[..., 1:4], Q[..., 4] = 1.0, (1.0, 2.0, 3.0), 5.0
        blk.Q.set(Q)
    exchange(mb, "Q")
    low, high = a.get_intface(5), b.get_intface(6)
    assert low.rotperiodic and high.rotperiodic
    # block 0's far end comes back through its low face turned by -90 about +x, (y, z) -> (z, -y); block 1's through its high face by +90
    for g in range(ng):
        assert np.allclose(a.Q.get()[ng, ng, g, 1:4], (1.0, 3.0, -2.0), rtol=1e-15, atol=1e-15)
        assert np.allclose(b.Q.get()[ng, ng, -1 - g, 1:4], (1.0, -3.0, 2.0), rtol=1e-15, atol=1e-15)
    # which is what the face's own rot does to the vector
    assert np.allclose(a.Q.get()[ng, ng, 0, 1:4], low.rot @ np.array([1.0, 2.0, 3.0]), rtol=1e-15, atol=1e-15)
    # a scalar is carried as it is
    assert a.Q.get()[ng, ng, 0, 0] == 1.0 and a.Q.get()[ng, ng, 0, 4] == 5.0
    # only the periodic faces' entries carry a rot
    c = mb.interiorcoll
    assert [c.rot(i) is not None for i in c.entries] == [i.rotperiodic for i in c.entries]


@pytest.mark.parametrize("field", [radial, azimuthal])
def test_a_rotational_periodic_lands_a_field_at_the_halos_centers(backend, recon, field):
    # a wrong sign or angle in the rot would land the field somewhere other than the halo's own centers
    mb = annulus(backend, recon)
    for blk in mb.blocks:
        Q = blk.Q.get()
        Q[..., 0], Q[..., 4] = 1.0, 5.0
        Q[..., 1:4] = field(blk.xcc.get())
        blk.Q.set(Q)
    exchange(mb, "Q")
    checked = 0
    for blk in mb.blocks:
        for face in blk.intfaces:
            if not face.rotperiodic:
                continue
            # the node exchange has already turned the halo's nodes into place, so its own centers are the reference
            halo, centers = face.halo(blk.Q.get()), face.halo(blk.xcc.get())
            assert np.allclose(halo[..., 1:4], field(centers), rtol=0, atol=1e-14)
            assert (halo[..., 0] == 1.0).all() and (halo[..., 4] == 5.0).all()
            checked += 1
    assert checked == 2


def test_a_rotational_periodics_halo_continues_the_sweep(backend, recon):
    # the centers step by the same 15 degrees through halo and interior, so the halo is the far end turned back by the sweep
    mb = annulus(backend, recon)
    ng = mb.ng
    x = mb.blocks[0].xcc.get()[ng, ng]
    angles = np.degrees(np.arctan2(x[:, 2], x[:, 1]))
    assert np.allclose(angles, -7.5 - 15 * (ng - 1) + 15 * np.arange(3 + 2 * ng), atol=1e-12)


def test_the_entrys_views_are_the_blocks_planes(backend, recon):
    # the halo view's layer 0 is nearest the face and the interior view runs inward from it
    mb = pair(backend, recon)
    stamp(mb, "Q")
    ng = mb.ng
    c = mb.interiorcoll
    for i, halos, interiors in zip(c.entries, c.column("Q"), c.column(("interior", "Q"))):
        Q = i.blk.Q.get()
        planes = np.moveaxis(Q, i.axis, 0)[:, ng:-ng, ng:-ng]
        halo = planes[ng - 1 :: -1] if i.is_low else planes[-ng:]
        interior = planes[ng : 2 * ng] if i.is_low else planes[-ng - 1 : -2 * ng - 1 : -1]
        assert np.array_equal(halos.get(), halo)
        assert np.array_equal(interiors.get(), interior)
    # the neighbour's planes seen in our frame are what the fill put in our halo
    exchange(mb, "Q")
    for halos, across in zip(c.column("Q"), c.column(("nbr", "Q"))):
        assert np.array_equal(halos.get(), across.get())
