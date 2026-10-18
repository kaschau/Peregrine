import numpy as np
import pytest

import peregrine as pg
from peregrine.multiblock.interface import BaseInterface


def mesher():
    return pg.mesher.CubeMesher(
        mbdims=[1, 1, 1], blkdims=[5, 4, 3], lengths=[1.0, 0.5, 0.25]
    )


class Jostled(pg.mesher.CubeMesher):
    def shape_blk(self, blk, i, j, k):
        super().shape_blk(blk, i, j, k)
        rng = np.random.default_rng(11)
        nodes = blk.nodes.get()
        ng = blk.ng
        # the boundary planes stay, so the volumes still sum to the box
        inner = tuple(slice(ng + 1, -ng - 1) for _ in range(3))
        nodes[inner] += 0.02 * rng.uniform(-1, 1, nodes[inner].shape)
        blk.nodes.set(nodes)


def grid(mesh, ng):
    mb = pg.multiblock.Grid(ng)
    mesh.fill(mb)
    if ng:
        mb.gen_halo()
    mb.make_metrics()
    return mb


def metrics(blk):
    m = {name: getattr(blk, name).get() for name in ("iS", "jS", "kS", "ixc", "jxc", "kxc", "xcc", "Jinv", "dijk")}
    m["V"] = 1.0 / blk.detJinv.get()
    spans = (m["ixc"][1:] - m["ixc"][:-1], m["jxc"][:, 1:] - m["jxc"][:, :-1], m["kxc"][:, :, 1:] - m["kxc"][:, :, :-1])
    m["spans"] = np.stack(spans, axis=-1)
    return m


def closure(m):
    iS, jS, kS = m["iS"], m["jS"], m["kS"]
    return (iS[1:] - iS[:-1]) + (jS[:, 1:] - jS[:, :-1]) + (kS[:, :, 1:] - kS[:, :, :-1])


def upindex(m):
    xcc = m["xcc"]
    return (
        (m["iS"][1:-1] * (xcc[1:] - xcc[:-1])).sum(-1),
        (m["jS"][:, 1:-1] * (xcc[:, 1:] - xcc[:, :-1])).sum(-1),
        (m["kS"][:, :, 1:-1] * (xcc[:, :, 1:] - xcc[:, :, :-1])).sum(-1),
    )


def test_a_haloed_grid_holds_a_solvers_metrics():
    ng = 2
    mb = grid(mesher(), ng)
    blk = mb.blocks[0]
    nodes = blk.nodes.get()
    assert nodes.shape == (5 + 2 * ng, 4 + 2 * ng, 3 + 2 * ng, 3)
    # the interior is the mesh; the halo continues it
    bare = pg.multiblock.Grid()
    mesher().fill(bare)
    assert np.array_equal(nodes[ng:-ng, ng:-ng, ng:-ng], bare.blocks[0].nodes.get())
    assert nodes[0, ng, ng, 0] == pytest.approx(-2.0 * 0.25)
    # the metrics cover the halo too, finite everywhere
    assert blk.iS.get().shape[0] == 5 + 2 * ng
    assert np.all(np.isfinite(blk.dijk.get()))


def test_a_tiny_cube_has_its_exact_volume():
    # cells of 2.5e-6 m: a volume of 1.6e-17 m^3, well under any floor
    mb = grid(pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[5, 5, 5], lengths=[1e-5] * 3), 2)
    blk = mb.blocks[0]
    V = 1.0 / blk.detJinv.get()[blk.interior]
    assert np.allclose(V, 2.5e-6**3, rtol=1e-12, atol=0)
    assert np.allclose(blk.dijk.get()[blk.interior], 2.5e-6, rtol=1e-12, atol=0)


def test_a_tiny_annulus_cell_is_its_prism():
    # a wedge cell of planar faces is a prism: axial length times the trapezoid 0.5 (r2^2 - r1^2) sin(dtheta), 1.2e-17 m^3 here
    mesh = pg.mesher.AnnulusMesher(mbdims=[1, 1, 1], blkdims=[5, 5, 5], p1=[0, 0, 0], p2=[1e-5, 0, 0], p3=[0, 1e-4, 0], sweep=4.5, thickness=1e-5)
    blk = grid(mesh, 1).blocks[0]
    V = 1.0 / blk.detJinv.get()[blk.interior]
    r1 = 1e-4 + 2.5e-6 * np.arange(4)
    prism = 2.5e-6 * 0.5 * ((r1 + 2.5e-6) ** 2 - r1**2) * np.sin(np.radians(4.5 / 4))
    assert np.allclose(V, prism[None, :, None], rtol=1e-13, atol=0)


def test_the_metrics_of_an_annulus_obey_the_laws():
    # a quarter annulus of r 0.5..1, length 1, 5 x 4 x 6 cells of 15 degrees
    mesh = pg.mesher.AnnulusMesher(mbdims=[1, 1, 1], blkdims=[6, 5, 7], p1=[0, 0, 0], p2=[1, 0, 0], p3=[0, 0.5, 0], sweep=90, thickness=0.5)
    blk = grid(mesh, 1).blocks[0]
    inner = blk.interior
    m = metrics(blk)
    scale = np.linalg.norm(m["iS"], axis=-1).max()
    # every cell closes
    assert np.abs(closure(m)[inner]).max() < 1e-14 * scale
    # the volume is the planar-faced wedge's, 0.5 (r2^2 - r1^2) sin(dtheta) per cell and length
    chord = 0.5 * (1.0**2 - 0.5**2) * np.sin(np.radians(15)) * 6 * 1.0
    assert m["V"][inner].sum() == pytest.approx(chord, rel=1e-12)
    # Jinv inverts the spans on these prism cells, and its determinant is the volume's inverse
    product = np.einsum("...ij,...jk->...ik", m["Jinv"], m["spans"])
    assert np.abs(product[inner] - np.eye(3)).max() < 1e-12
    assert np.abs(np.linalg.det(m["Jinv"][inner]) * m["V"][inner] - 1.0).max() < 1e-12
    # the area vectors point up their index
    assert all(s[inner].min() > 0 for s in upindex(m))
    # the lengths are the chords: axial, radial across the 7.5 degree half angle, and around at the cell's mid radius
    rmid = 0.5 + 0.125 * (np.arange(4) + 0.5)
    expected = (1.0 / 5, 0.125 * np.cos(np.radians(7.5)), 2 * rmid * np.sin(np.radians(7.5)))
    d = m["dijk"][inner]
    assert np.allclose(d[..., 0], expected[0], rtol=1e-14, atol=0)
    assert np.allclose(d[..., 1], expected[1], rtol=1e-14, atol=0)
    assert np.allclose(d[..., 2], expected[2][None, :, None], rtol=1e-14, atol=0)


def test_the_metrics_of_a_jostled_cube_obey_the_laws():
    # on non-planar faces Jinv is the adjugate of the spans over the face-based volume, which the edge jacobian's determinant is not
    blk = grid(Jostled(mbdims=[1, 1, 1], blkdims=[7, 6, 5], lengths=[1, 1, 1]), 1).blocks[0]
    inner = blk.interior
    m = metrics(blk)
    scale = np.linalg.norm(m["iS"], axis=-1).max()
    assert np.abs(closure(m)[inner]).max() < 1e-14 * scale
    assert m["V"][inner].sum() == pytest.approx(1.0, rel=1e-13)
    product = np.einsum("...ij,...jk->...ik", m["Jinv"], m["spans"])
    adjugate = (np.linalg.det(m["spans"]) / m["V"])[..., None, None] * np.eye(3)
    assert np.abs(product[inner] - adjugate[inner]).max() < 1e-13
    assert np.abs(product[inner] - np.eye(3)).max() > 1e-2
    assert all(s[inner].min() > 0 for s in upindex(m))


def test_a_cell_turned_inside_out_is_refused():
    mb = pg.multiblock.Grid(1)
    mesher().fill(mb)
    blk = mb.blocks[0]
    nodes = blk.nodes.get()
    # swap two node planes along i, inverting the cells between them
    nodes[[2, 3]] = nodes[[3, 2]]
    blk.nodes.set(nodes)
    mb.gen_halo()
    with pytest.raises(ValueError, match="no volume or a negative one"):
        mb.make_metrics()


# an axis with no more planes than the halo is deep cannot reflect, so gen_halo continues its two boundary-nearest planes instead
def reflected(interior, ng, thick):
    out = interior
    for a in range(3):
        if thick[a]:
            pad = [(0, 0)] * 4
            pad[a] = (ng, ng)
            out = np.pad(out, pad, mode="reflect", reflect_type="odd")
    for a in range(3):
        if not thick[a]:
            x = np.moveaxis(out, a, 0)
            low = [(1 + n) * x[0] - n * x[1] for n in range(ng, 0, -1)]
            high = [(1 + n) * x[-1] - n * x[-2] for n in range(1, ng + 1)]
            out = np.moveaxis(np.concatenate([np.stack(low), x, np.stack(high)]), 0, a)
    return out


@pytest.mark.parametrize("ng,dims", [(2, [5, 4, 3]), (2, [5, 4, 2]), (3, [5, 3, 2])])
def test_the_node_halo_reflects_the_interior(ng, dims):
    # jostled, so extrapolation schemes differ and only the odd reflection passes
    mb = pg.multiblock.Grid(ng)
    Jostled(mbdims=[1, 1, 1], blkdims=dims, lengths=[1.0, 0.5, 0.25]).fill(mb)
    blk = mb.blocks[0]
    interior = blk.nodes.get()[blk.interior].copy()
    mb.gen_halo()
    expected = reflected(interior, ng, [n > ng for n in dims])
    assert np.allclose(blk.nodes.get(), expected, rtol=0, atol=1e-14)


def forget_periodics(mb):
    cleared = {}
    for blk, face in mb.intfaces():
        if face.rot is not None:
            cleared[blk.nblki, face.nface] = (face.rot.copy(), face.xlate.copy())
            face.set_interior(face.nbr, face.orient)
    return cleared


def test_a_translated_periodic_is_found_from_the_planes():
    mb = pg.multiblock.Grid()
    pg.mesher.CubeMesher(mbdims=[2, 1, 1], blkdims=[4, 3, 3], lengths=[2, 1, 1], periodic=[True] * 3).fill(mb)
    cleared = forget_periodics(mb)
    # the i wrap of each block and the j and k wraps of both: ten
    assert len(cleared) == 10
    assert mb.mark_periodics() == {"periodic-xlate": 10}
    for (nblki, nface), (rot, xlate) in cleared.items():
        face = mb.get_blk(nblki).get_intface(nface)
        assert np.allclose(face.rot, rot, atol=1e-12) and np.allclose(face.xlate, xlate, atol=1e-12)
    # found once, nothing is found again
    assert mb.mark_periodics() == {}


def test_a_rotational_periodic_is_found_from_the_planes():
    mb = pg.multiblock.Grid()
    pg.mesher.AnnulusMesher(mbdims=[1, 1, 2], blkdims=[4, 4, 4], p1=[0, 0, 0], p2=[1, 0, 0], p3=[0, 0.5, 0], sweep=90, thickness=0.5, periodic=True, defaultbc="wall").fill(mb)
    cleared = forget_periodics(mb)
    assert len(cleared) == 2
    assert mb.mark_periodics() == {"periodic-rot": 2}
    for (nblki, nface), (rot, xlate) in cleared.items():
        face = mb.get_blk(nblki).get_intface(nface)
        assert np.allclose(face.rot, rot, atol=1e-12) and np.allclose(face.xlate, 0, atol=1e-12)
    assert np.allclose(mb.get_blk(0).get_intface(5).rot, BaseInterface.rotabout([1, 0, 0], -90), atol=1e-12)


def test_a_plane_is_joined_or_a_rigid_move_apart():
    rng = np.random.default_rng(3)
    mine = rng.random((12, 3))
    assert pg.multiblock.Grid._xform(mine, mine, 1e-8) is None
    rot, xlate = pg.multiblock.Grid._xform(mine - [0.0, 2.0, 0.0], mine, 1e-8)
    assert np.allclose(rot, np.eye(3)) and np.allclose(xlate, [0.0, 2.0, 0.0])
    # theirs is mine turned back by 30 degrees, so the rot carrying theirs onto mine is the turn
    turn = BaseInterface.rotabout([0, 0, 1], 30)
    rot, xlate = pg.multiblock.Grid._xform(mine @ turn, mine, 1e-8)
    assert np.allclose(rot, turn, atol=1e-12) and np.allclose(xlate, 0)
    # a plane sheared off its partner is neither
    sheared = mine * [1.0, 1.0, 1.1]
    with pytest.raises(ValueError, match="neither joined nor a rigid transform"):
        pg.multiblock.Grid._xform(sheared, mine, 1e-8)


def test_a_grid_without_a_halo_refuses_to_make_one():
    mb = pg.multiblock.Grid()
    mesher().fill(mb)
    before = mb.blocks[0].nodes.get().copy()
    with pytest.raises(ValueError, match="no halo"):
        mb.gen_halo()
    assert np.array_equal(mb.blocks[0].nodes.get(), before)
