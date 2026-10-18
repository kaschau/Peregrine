import numpy as np
import pytest

import peregrine as pg

from tests.gases import ondevice
from tests.minimal.cases import FLUXES, advflux, box, derivative, fluxid


# a uniform state's faces flux alike and the cube's opposite faces are alike bitwise, so nothing is left for a tolerance to hide
@pytest.mark.parametrize("keys", FLUXES, ids=fluxid)
def test_rest_stays_exactly(backend, keys):
    config, mesh = box()
    advflux(config, **keys)
    mb = pg.get_solver(config, mesh, backend(config))
    # rusanov's dissipation cancels the flux difference to the bit on the host; a device's fused multiply-add leaves 1e-25
    if keys["primary"] == "rusanov" and ondevice(mb):
        pytest.skip("rusanov's cancellation at rest is bitwise on the host alone")
    blk = mb.blocks[0]
    Q0 = blk.Q.get()
    mb.step(config["simulation"]["dt"])
    assert np.array_equal(blk.Q.get(), Q0)
    assert mb.nrt == 1
    # sharper than the step can show, under the solution's own ulp: the derivative itself
    assert not derivative(mb).any()
