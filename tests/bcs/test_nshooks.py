import numpy as np
import pytest

from tests.bcs.bcblock import assert_consistent, case

INLETS_AND_EXITS = (
    "sub-in-fvt",
    "sup-in-fa",
    "sub-in-ftptt",
    "sub-in-fmt",
    "sub-out-fp",
    "sup-out-fn",
)


@pytest.mark.parametrize("kind", INLETS_AND_EXITS)
def test_the_gradients_are_the_images(backend, kind):
    mb = case(backend, kind, "kepaec", "CH4_O2", simulator="navier-stokes")
    blk = mb.blocks[0]
    mb.apply_bcs("preadvflux")
    assert_consistent(mb, blk)
    before = blk.Q.get().copy()
    mb.apply_bcs("pregrads")
    assert np.array_equal(blk.Q.get(), before)
    rng = np.random.default_rng(5)
    grads = rng.uniform(-1.0, 1.0, blk.grads.get().shape)
    blk.grads.set(grads)
    mb.apply_bcs("postgrads")
    grads = blk.grads.get()
    ng = blk.ng
    inner = np.s_[ng:-ng, ng:-ng]
    for interface in blk.intfaces:
        planes = np.moveaxis(grads, interface.axis, 0)[ng:-ng]
        images = planes if interface.is_low else planes[::-1]
        for halo, image in zip(interface.halo(grads), images):
            assert np.allclose(halo[inner], image[inner])
