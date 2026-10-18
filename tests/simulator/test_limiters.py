import numpy as np
import pytest

import peregrine as pg
from peregrine.launch.kernel import Kernel
from peregrine.launch.ranges import FaceRange

from tests.simulator import oracle as orc
from tests.simulator import support as sp

symmetric = ("minmod", "mc", "superbee", "vanleer")
# against the slope, flat, under one, one, over one and far past the limit
ratios = np.array([-1e3, -2.0, -1.0, -0.5, -1e-9, 0.0, 1e-9, 0.1, 0.25, 0.5, 2.0 / 3.0, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 1e3, 1e9])


class WithPhi(pg.multiblock.Solver):
    def _decl_arrays(self):
        super()._decl_arrays()
        self.decl_array("ratio", FaceRange, comps=1, axis=0)
        self.decl_array("phi", FaceRange, comps=1, axis=0)

    def _decl_kerns(self):
        super()._decl_kerns()
        tpl = f"{__package__}.kernels.limiter"
        self.kernels["limiter"] = lambda: Kernel(self.provider, tpl, axis=0, options=dict(self.cfg["adv-flux"]), ratio="ratio", out="phi")


def phi(backend, limiter, r):
    mb = sp.line(backend, len(r) + 1, gas=sp.air, cls=WithPhi, primary="rusanov", recon="muscl", limiter=limiter)
    blk = mb.blocks[0]
    ng = blk.ng
    ratio = blk.ratio.get()
    ratio[ng : ng + len(r), ng, ng, 0] = r
    blk.ratio.set(ratio)
    mb.launch(mb.kernels["limiter"](), "interior")
    return blk.phi.get()[ng : ng + len(r), ng, ng, 0]


@pytest.mark.parametrize("limiter", orc.limiters)
def test_phi_is_its_formula(backend, limiter):
    # the same operations in the same order: measured bitwise
    assert np.array_equal(phi(backend, limiter, ratios), orc.limiters[limiter](ratios))


@pytest.mark.parametrize("limiter", orc.limiters)
def test_phi_lies_in_swebys_region(backend, limiter):
    # none against the slope or on a flat upwind, at most twice the ratio and at most two elsewhere, one at one
    got = phi(backend, limiter, ratios)
    positive = ratios > 0.0
    assert (got[~positive] == 0.0).all()
    assert (got[positive] >= 0.0).all()
    assert (got[positive] <= np.minimum(2.0 * ratios[positive], 2.0)).all()
    assert got[ratios == 1.0] == 1.0


@pytest.mark.parametrize("limiter", orc.limiters)
def test_the_symmetric_limiters_treat_the_two_slopes_alike(backend, limiter):
    # phi(r) / r = phi(1 / r): the fraction of either slope the face moves along is the same whichever is the ratio's numerator; Koren's bias toward the upwind slope breaks it
    r = ratios[ratios > 0.0]
    own, swapped = phi(backend, limiter, r) / r, phi(backend, limiter, 1.0 / r)
    if limiter in symmetric:
        # measured 2.2e-16, minmod and superbee exact
        assert np.allclose(own, swapped, rtol=1e-14, atol=0.0)
    else:
        assert not np.allclose(own, swapped, rtol=1e-2, atol=0.0)
