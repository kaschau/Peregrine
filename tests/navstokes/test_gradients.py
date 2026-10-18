import numpy as np

import peregrine as pg

from tests.navstokes import support

# the gradient of every quantity the kernel takes one of: du_i/dx_j, dT/dx and dY/dx
A = np.array([[3.0, -2.0, 1.0], [0.5, 4.0, -1.5], [-2.0, 1.0, 2.5]])
aT = np.array([20.0, -30.0, 10.0])
aY = np.array([0.05, 0.1, -0.08])


def solver(backend, mesh):
    config = support.config(support.alike)
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    return pg.get_solver(config, mesh, backend(config))


def errors(mb):
    blk = mb.blocks[0]
    ng = blk.ng
    x = blk.xcc.get()
    support.setprims(mb, blk, 101325.0, 10.0 + x @ A.T, 300.0 + x @ aT, (0.6 + x @ aY)[..., None])
    mb.launch(mb.kernels["dqdxyz"]("Q"), "interior")
    # the walls mirror the halo, so a cell beside it does not see the linear field
    deep = np.s_[ng + 1 : -ng - 1, ng + 1 : -ng - 1, ng + 1 : -ng - 1]
    want = np.concatenate([A, aT[None], aY[None]])
    got = blk.grads.get()[deep]
    return np.abs(got - want).max(axis=(0, 1, 2)) / np.abs(want).max(axis=1)[:, None]


def test_linear_fields_are_exact_on_a_cartesian_box(backend):
    # measured 2.3e-14 over every row
    mb = solver(backend, support.box([9, 8, 7], [1.0, 1.0, 1.0], defaultbc="walls"))
    assert errors(mb).max() < 1e-12


def test_second_order_under_a_smooth_mapping(backend):
    errs = []
    for n in (9, 17, 33):
        mb = solver(backend, support.SmoothCube(blkdims=[n, n, n], defaultbc="walls"))
        errs.append(errors(mb).max())
    assert np.log2(errs[1] / errs[2]) >= 1.8
    assert errs[2] < 6e-3


# Jinv inverts the cell's mean-edge jacobian (block/grid.py) while the kernel differences neighbouring centres: the two agree only where the mapping from index space is smooth
def test_no_convergence_on_a_jostled_grid(backend):
    errs = []
    for n in (9, 17, 33):
        mesh = support.SmoothCube(jostle=0.1, blkdims=[n, n, n], defaultbc="walls")
        errs.append(errors(solver(backend, mesh)).max())
    assert all(0.1 < e < 0.4 for e in errs), errs
