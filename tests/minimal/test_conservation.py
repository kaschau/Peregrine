import numpy as np
import pytest

import peregrine as pg

from tests.gases import primitives
from tests.minimal.cases import (
    BLENDED,
    FLUXES,
    JostledCube,
    advflux,
    box,
    derivative,
    fluxid,
    fluxscale,
    volumes,
    wallfluxes,
)

# the adiabatic wall of each physics; the viscous one holds the flow
WALL = {"euler": "slp-adia-wall", "navier-stokes": "no-slp-adia-wall"}

pytestmark = [
    pytest.mark.parametrize("physics", ["euler", "navier-stokes"]),
    pytest.mark.parametrize("keys", FLUXES + [BLENDED], ids=fluxid),
]


def solver(backend, physics, periodic, **keys):
    config, mesh = box(dims=(8, 7, 6), periodic=(periodic,) * 3, physics=physics, mesher=JostledCube)
    if not periodic:
        config["bcvals"]["wall"] = {"kind": WALL[physics]}
    advflux(config, **keys)
    mb = pg.get_solver(config, mesh, backend(config))
    blk = mb.blocks[0]
    rng = np.random.default_rng(3)
    q = primitives(mb, blk)
    shape = q.shape[:3]
    q[..., 0] = 101325.0 * (1.0 + 0.1 * rng.uniform(-1.0, 1.0, shape))
    q[..., 1:4] = 60.0 * rng.uniform(-1.0, 1.0, shape + (3,))
    q[..., 4] = 300.0 * (1.0 + 0.1 * rng.uniform(-1.0, 1.0, shape))
    q[..., 5] = 0.77 + 0.2 * rng.uniform(-1.0, 1.0, shape)
    mb.set_prims([q])
    return mb


def budget(mb):
    blk = mb.blocks[0]
    net = np.einsum("ijkl,ijk->l", derivative(mb), volumes(blk))
    scale = fluxscale(blk)
    assert scale.min() > 0.0
    return net, scale


def test_a_periodic_box_telescopes(backend, physics, keys):
    net, scale = budget(solver(backend, physics, True, **keys))
    # measured within 4.2e-15
    assert (np.abs(net) <= 1e-13 * scale).all()


def test_a_walled_box_takes_only_what_crosses_its_walls(backend, physics, keys):
    mb = solver(backend, physics, False, **keys)
    net, scale = budget(mb)
    walls = wallfluxes(mb.blocks[0])
    # the sum is the boundary fluxes: measured within 2.2e-14
    assert (np.abs(net - walls) <= 1e-13 * scale).all()
    # nothing but momentum crosses a wall: measured within 2.6e-15
    rows = [0, 4, 5]
    assert (np.abs(walls[rows]) <= 1e-13 * scale[rows]).all()
    # and the walls push: the momentum rows are the pressure forces, measured at 4 to 97 percent of the largest flux
    assert (np.abs(walls[1:4]) > 1e-3 * scale[1:4]).all()
