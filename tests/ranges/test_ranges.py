import numpy as np
import pytest

import peregrine as pg
from peregrine.launch.policy import FacePolicy
from peregrine.launch.ranges import CellCenterRange, FaceRange, NodeRange

extents, ng = (8, 7, 6), 2


def block():
    mb = pg.multiblock.Grid(ng=ng)
    pg.mesher.CubeMesher(mbdims=[2, 1, 1], blkdims=list(extents), lengths=[2, 1, 1], defaultbc="wall").fill(mb)
    return mb.get_blk(0)


def kinds(blk):
    yield "cells", CellCenterRange(blk)
    yield "nodes", NodeRange(blk)
    for axis in range(3):
        yield f"faces {axis}", FaceRange(blk, axis)


def marked(rng, ranges):
    hit = np.zeros(rng.fullexts, dtype=int)
    for start, extent in ranges:
        hit[tuple(slice(s, s + e) for s, e in zip(start, extent))] += 1
    return hit


@pytest.fixture(scope="module")
def blk():
    return block()


@pytest.fixture(params=["cells", "nodes", "faces 0", "faces 1", "faces 2"])
def rng(request, blk):
    return dict(kinds(blk))[request.param]


def test_all_is_interior_and_six_disjoint_halos(blk, rng):
    hit = marked(rng, rng.all())
    assert hit.max() == 1
    assert hit.sum() == np.prod(rng.exts) + sum(
        np.prod(e) for face in blk.intfaces for _, e in rng.halo(face)
    )
    # every interior index once
    assert (marked(rng, rng.interior()) == 1).sum() == np.prod(rng.exts)


def test_shallow_is_the_interior_and_the_first_layer_of_every_halo(blk, rng):
    hit = marked(rng, rng.shallow())
    assert hit.max() == 1
    assert hit.sum() == np.prod(rng.exts) + sum(
        np.prod(e) for face in blk.intfaces for _, e in rng.halo(face, 1)
    )
    # within all, the interior whole, and of each halo the layer nearest the interface alone
    assert (marked(rng, rng.all()) >= hit).all()
    assert (marked(rng, rng.interior()) <= hit).all()
    for face in blk.intfaces:
        for layer in range(ng):
            ((start, extent),) = rng.halolayer(face, layer)
            inside = hit[tuple(slice(s, s + e) for s, e in zip(start, extent))]
            assert inside.all() if layer == 0 else not inside.any()


def test_no_halo_edge_or_corner_is_ever_iterated(rng):
    hit = marked(rng, rng.all())
    lo, hi = ng, [n - ng for n in rng.fullexts]
    # a halo edge or corner index is outside the interior along two or more axes
    idx = np.indices(rng.fullexts)
    outside = sum((idx[a] < lo) | (idx[a] >= hi[a]) for a in range(3))
    assert not hit[outside >= 2].any()
    assert hit[outside == 0].all() and hit[outside == 1].all()


def test_full_covers_everything(rng):
    assert (marked(rng, rng.full()) == 1).all()
    assert rng.fullexts == tuple(n + 2 * ng for n in rng.exts)


def test_halo_depth_and_extents(blk):
    rng = CellCenterRange(blk)
    for face in blk.intfaces:
        for depth in (1, 2):
            ((start, extent),) = rng.halo(face, depth)
            axis = face.axis
            assert extent[axis] == depth
            # against the interface proper, not the interior's far side
            assert start[axis] == (ng - depth if face.is_low else ng + rng.exts[axis])
            layer, a, b = rng.haloexts(face, depth)
            assert layer == depth and (a, b) == tuple(e for d, e in enumerate(extent) if d != axis)


def cellsof(origin, axes, extents):
    cells = np.empty(tuple(extents) + (3,), dtype=int)
    for g, a, b in np.ndindex(*extents):
        cell = list(origin)
        for d, n in zip(axes, (g, a, b)):
            cell[abs(d) - 1] += n if d > 0 else -n
        cells[g, a, b] = cell
    return cells


@pytest.mark.parametrize("kind", [CellCenterRange, NodeRange])
def test_the_maps_say_what_the_layers_do(blk, kind):
    # the halo map's layer g is halolayer(g); the interior map's layer g is the plane g past the start, counted inward from the face
    rng = kind(blk)
    for face in blk.intfaces:
        extents = rng.haloexts(face, ng)
        halo = cellsof(*rng.halomap(face), extents)
        interior = cellsof(*rng.interiormap(face), extents)
        axis = face.axis
        for g in range(ng):
            ((start, extent),) = rng.halolayer(face, g)
            layer = halo[g]
            assert layer[..., axis].min() == layer[..., axis].max() == start[axis]
            inward = 1 if face.is_low else -1
            # the interior plane g lies 2g + 1 + skip cells inward of halo layer g
            assert (interior[g][..., axis] == start[axis] + inward * (2 * g + 1 + rng.xchgstart)).all()
            # the plane itself is the interface proper, in axis order
            plane = [d for d in range(3) if d != axis]
            for m, d in enumerate(plane):
                assert layer[..., d].min() == ng and layer[..., d].max() == ng + extents[1 + m] - 1
                assert np.array_equal(layer[..., d], interior[g][..., d])


def test_nodes_exchange_starts_one_plane_in(blk):
    assert NodeRange(blk).xchgstart == 1
    assert CellCenterRange(blk).xchgstart == 0
    for axis in range(3):
        assert FaceRange(blk, axis).xchgstart == 0


def test_cell_face_kinds_have_one_more_along_their_axis(blk):
    cells = CellCenterRange(blk).exts
    for axis in range(3):
        faces = FaceRange(blk, axis).exts
        assert all(faces[a] == cells[a] + (a == axis) for a in range(3))
        for nface in (2 * axis + 1, 2 * axis + 2):
            face = blk.get_intface(nface)
            ((start, extent),) = FaceRange(blk, axis).plane(face)
            assert extent[axis] == 1
            assert start[axis] == (ng if face.is_low else ng + faces[axis] - 1)
        with pytest.raises(AssertionError):
            FaceRange(blk, axis).plane(blk.get_intface(2 * ((axis + 1) % 3) + 1))


def test_a_range_is_cut_at_given_interfaces():
    # the middle block of a [3, 2, 1] lattice meets neighbors on faces 1, 2 and 4; cut at the low i and the high j
    mb = pg.multiblock.Grid(ng=ng)
    pg.mesher.CubeMesher(mbdims=[3, 2, 1], blkdims=list(extents), lengths=[3, 2, 1], defaultbc="wall").fill(mb)
    blk = mb.get_blk(1)
    assert [f.nface for f in blk.intfaces if f.nbr is not None] == [1, 2, 4]
    at = [(blk.get_intface(1), None), (blk.get_intface(4), None)]
    cells = CellCenterRange(blk)
    for name in ("all", "halos", "shallow"):
        named = getattr(cells, name)()
        less, behind = cells.less(name, at), cells.behind(name, at)
        # the two parts are the named range, once each
        assert sorted(less + behind) == sorted(named)
        hit, whole = marked(cells, less), marked(cells, named)
        assert hit.max() == 1
        # nothing behind a cut interface, whatever the range held there; behind the others, what it holds
        for f in blk.intfaces:
            ((start, extent),) = cells.halo(f)
            slab = tuple(slice(s, s + e) for s, e in zip(start, extent))
            if f in [i for i, _ in at]:
                assert not hit[slab].any()
            else:
                assert np.array_equal(hit[slab], whole[slab])
    # the interior holds no slab: whole, with nothing behind
    assert cells.less("interior", at) == cells.interior() and cells.behind("interior", at) == []
    # what lies behind is as deep as said, and no deeper than the range holds
    assert cells.behind("halos", [(blk.get_intface(1), 1)]) == cells.halo(blk.get_intface(1), 1)
    assert cells.behind("shallow", [(blk.get_intface(1), None)]) == cells.halo(blk.get_intface(1), 1)
    assert cells.behind("shallow", [(blk.get_intface(1), ng)]) == cells.halo(blk.get_intface(1), 1)
    # the faces of an axis lose only the plane on a given interface of theirs: the low plane of axis 0, the high plane of axis 1, none of axis 2
    for axis in range(3):
        faces = FaceRange(blk, axis)
        left = marked(faces, faces.interior()) - marked(faces, faces.less("interior", at))
        assert left.min() == 0
        planes = faces.behind("interior", at)
        assert planes == [p for f, _ in at if f.axis == axis for p in faces.plane(f)]
        assert np.array_equal(left, marked(faces, planes))
        ((start, extent),) = faces.less("interior", at)
        assert extent[axis] == faces.exts[axis] - (axis < 2)
        assert start[axis] == ng + (axis == 0)
        with pytest.raises(ValueError):
            faces.less("full", at)


def test_a_face_plane_on_an_interface_carries_the_interior_cell(blk):
    for intface in blk.intfaces:
        policy = FacePolicy(intface.axis)
        policy.carried = [policy.offset("interior")]
        [(start, extent, [behind])] = policy.haloranges(intface)
        before = [0, 0, 0]
        before[intface.axis] = -1
        assert behind == ((0, 0, 0) if intface.is_low else tuple(before))
        assert extent[intface.axis] == 1
