import numpy as np
import pytest

import peregrine as pg

from tests.gases import primitives
from tests.minimal.cases import BLENDED, FLUXES, JostledCube, advflux, box, derivative, fluxid, fluxscale, volumes


@pytest.mark.parametrize("physics", ["euler", "navier-stokes"])
@pytest.mark.parametrize("keys", FLUXES + [BLENDED], ids=fluxid)
def test_a_uniform_flow_is_preserved(backend, physics, keys):
    config, mesh = box(dims=(8, 7, 6), periodic=(True, True, True), physics=physics, mesher=JostledCube)
    advflux(config, **keys)
    mb = pg.get_solver(config, mesh, backend(config))
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    # a velocity along every axis, so every row carries a flux to measure against
    q[..., :5] = (101325.0, 50.0, -30.0, 20.0, 300.0)
    mb.set_prims([q])
    net = derivative(mb) * volumes(blk)[..., None]
    scale = fluxscale(blk)
    assert scale.min() > 0.0
    # per row against the largest face flux of that row: measured within 7e-15
    assert (np.abs(net).max(axis=(0, 1, 2)) <= 1e-13 * scale).all()
