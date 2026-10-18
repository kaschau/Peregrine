import numpy as np

import peregrine as pg
from peregrine.tools.verify import verify

from tests.gases import configure
from tests.partition.test_cut import mutual


def test_faces_are_wired_and_named():
    mb = pg.multiblock.Grid()
    pg.mesher.CubeMesher(
        mbdims=[2, 1, 1],
        blkdims=[4, 3, 3],
        lengths=[2, 1, 1],
        periodic=[False, True, False],
        bcnames={1: "inlet", 6: "top"},
    ).fill(mb)
    a, b = mb.blocks
    assert a.get_intface(2).nbr == 1 and b.get_intface(1).nbr == 0
    assert a.get_intface(2).intkind == "interior" and a.get_intface(2).bcname is None
    # periodic in j: each block's j sides meet themselves, moved by the length
    for blk in mb.blocks:
        for nface in (3, 4):
            face = blk.get_intface(nface)
            assert face.nbr == blk.nblki and face.intkind == "periodic-xlate"
            assert np.allclose(np.abs(face.xlate), [0, 1, 0])
            assert np.allclose(face.rot, np.eye(3))
    assert a.get_intface(1).bcname == "inlet" and a.get_intface(1).intkind is None
    assert a.get_intface(6).bcname == "top" and b.get_intface(6).bcname == "top"
    assert a.get_intface(5).bcname is None and a.get_intface(5).intkind is None
    # the second block sits beside the first
    assert np.isclose(b.nodes.get()[..., 0].min(), a.nodes.get()[..., 0].max())


def test_a_named_side_keeps_its_name_over_the_default():
    mb = pg.multiblock.Grid()
    pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[3, 3, 3], bcnames={1: "inlet"}, defaultbc="wall").fill(mb)
    blk = mb.blocks[0]
    assert blk.get_intface(1).bcname == "inlet"
    assert [blk.get_intface(n).bcname for n in range(2, 7)] == ["wall"] * 5
    assert verify(mb)


def test_a_lattice_is_wired_along_every_axis():
    # block n = 6k + 2j + i of a [2, 3, 2] lattice: i neighbors one block apart, j two, k six, none past an edge
    mb = pg.multiblock.Grid()
    pg.mesher.CubeMesher(mbdims=[2, 3, 2], blkdims=[3, 3, 3], lengths=[2, 3, 2], defaultbc="wall").fill(mb)
    assert len(mb.blocks) == 12
    for blk in mb.blocks:
        n = blk.nblki
        i, j, k = n % 2, (n // 2) % 3, n // 6
        expected = {
            1: n - 1 if i > 0 else None,
            2: n + 1 if i < 1 else None,
            3: n - 2 if j > 0 else None,
            4: n + 2 if j < 2 else None,
            5: n - 6 if k > 0 else None,
            6: n + 6 if k < 1 else None,
        }
        for nface, nbr in expected.items():
            face = blk.get_intface(nface)
            assert face.nbr == nbr
            assert face.orient == ("123" if nbr is not None else None)
            assert face.bcname == (None if nbr is not None else "wall")
    assert mutual(mb)
    # the coordinates of every joined face coincide
    assert verify(mb)


def test_periodic_halos_are_the_far_side_moved(backend):
    config = pg.Config()
    configure(config, "air", "euler")
    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1],
        blkdims=[5, 4, 3],
        lengths=[2, 1, 1],
        periodic=[True, True, True],
    )
    mb = pg.get_solver(config, mesh, backend(config))
    blk = mb.blocks[0]
    ng = blk.ng
    nodes = blk.nodes.get()
    # the node halo past the high i side continues the grid with the same spacing
    dx = nodes[ng + 1, ng, ng, 0] - nodes[ng, ng, ng, 0]
    assert np.allclose(np.diff(nodes[:, ng, ng, 0]), dx)
    # and the cell centers likewise, both ends
    cells = blk.xcc.get()
    assert np.allclose(np.diff(cells[:, ng, ng, 0]), dx)
    assert np.allclose(cells[ng:-ng, ng, ng, 0].min(), dx / 2) and np.allclose(
        cells[0, ng, ng, 0], -dx / 2
    )
