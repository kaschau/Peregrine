import numpy as np

import peregrine as pg

from tests.gases import configure


# a block thinner than a boundary's reach reads its deepest cell, never the opposite halo
def slab(backend, cells, kind, recon="constant", **values):
    config = pg.Config()
    configure(config, "air", "euler")
    config["adv-flux"]["recon"] = recon
    if recon == "muscl":
        config["adv-flux"]["limiter"] = "koren"
    config["bcvals"]["thin"] = {"kind": kind, **values}
    config["bcvals"]["slip"] = {"kind": "slp-adia-wall"}
    mesh = pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[cells + 1, 6, 6], lengths=[1, 1, 1], bcnames={1: "thin", 2: "thin"}, defaultbc="slip")
    mb = pg.get_solver(config, mesh, backend(config))
    blk = mb.blocks[0]
    # a state varying along the slab's own axes, uniform across it
    prims = mb.sim.iovars.data(blk, mb.sim.iovars.privars)
    q = np.stack([prims[name] for name in mb.sim.iovars.privars], axis=-1)
    _, j, k = np.indices(q.shape[:3])
    q[..., 4] = 300.0 + 5.0 * j + 9.0 * k
    q[..., 1] = 10.0
    q[..., 2] = 20.0 + j
    mb.set_prims([q])
    mb.apply_bcs("preadvflux")
    return mb, blk


def test_a_one_cell_slab_between_walls_reflects_its_cell_into_every_layer(backend):
    mb, blk = slab(backend, 1, "slp-adia-wall", recon="muscl")
    assert mb.ng == 2
    Q = blk.Q.get()
    ng = blk.ng
    cell = Q[ng, ng:-ng, ng:-ng]
    for low, high in ((ng - 1, ng + 1), (ng - 2, ng + 2)):
        for halo in (Q[low, ng:-ng, ng:-ng], Q[high, ng:-ng, ng:-ng]):
            # the momentum reflected through the state's velocity, within rounding; the rest the cell's
            assert np.allclose(halo[..., 1], -cell[..., 1], rtol=1e-14, atol=0.0)
            assert np.array_equal(halo[..., [0, 2, 3, 4]], cell[..., [0, 2, 3, 4]])
    # the second layer is the first, both the one cell's reflection
    assert np.array_equal(Q[ng - 2], Q[ng - 1]) and np.array_equal(Q[ng + 2], Q[ng + 1])
    mb.step(1e-7)


def test_an_exit_on_a_one_cell_slab_extrapolates_from_its_cell(backend):
    mb, blk = slab(backend, 1, "sub-out-fp", p=101325.0)
    assert np.isfinite(blk.Q.get()).all()
    mb.step(1e-7)
