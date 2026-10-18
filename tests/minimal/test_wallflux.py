import numpy as np
import pytest

import peregrine as pg

from tests.gases import primitives
from tests.minimal.cases import FLUXES, advflux, box, fluxid, totals

WALLS = {
    "slp-adia-wall": {"kind": "slp-adia-wall"},
    "slp-isot-wall": {"kind": "slp-isot-wall", "T": 400.0},
}
WALLFLUXES = FLUXES


def cases():
    for keys in WALLFLUXES:
        for kind in WALLS:
            yield pytest.param(keys, kind, id=f"{fluxid(keys)}-{kind}")


def walled(backend, kind, into, **keys):
    dims, periodic = ((5, 6, 7), (False,) * 3) if into else ((6, 6, 7), (True, False, False))
    config, mesh = box(dims=dims, periodic=periodic)
    config["bcvals"]["wall"] = WALLS[kind]
    advflux(config, **keys)
    mb = pg.get_solver(config, mesh, backend(config))
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    i, j, k = np.indices(q.shape[:3])
    # the temperature varies along every axis, so an isothermal wall makes a density jump
    q[..., 4] = 300.0 + 5.0 * i + 7.0 * j + 9.0 * k
    q[..., 1] = 10.0 + i
    q[..., 2] = 20.0 + j if into else 0.0
    q[..., 3] = 30.0 + k if into else 0.0
    mb.set_prims([q])
    return mb


def firstcell(interface, array):
    ng = interface.ng
    planes = np.moveaxis(array, interface.axis, 0)
    plane = planes[ng] if interface.is_low else planes[len(planes) - ng - 1]
    return plane[ng:-ng, ng:-ng]


class WallFace:
    def __init__(self, mb, interface):
        blk = mb.blocks[0]
        fields = list(mb.sim.statefields)
        a = "ijk"[interface.axis]
        self.F = interface.faceplane(getattr(blk, f"{a}F").get())
        S = interface.faceplane(getattr(blk, f"{a}S").get())
        cell, state = firstcell(interface, blk.Q.get()), firstcell(interface, blk.state.get())
        self.rho = cell[..., 0]
        self.u = cell[..., 1:4] / self.rho[..., None]
        self.p, self.c, e = (state[..., fields.index(f)] for f in ("p", "c", "e"))
        self.area = np.sqrt(np.sum(S**2, axis=-1))
        self.n = S / self.area[..., None]
        self.un = np.sum(self.u * self.n, axis=-1)
        self.outward = -1.0 if interface.is_low else 1.0
        # a flow of the first cell's through the face, and a bound on the energy it would carry
        self.mass = self.rho * np.sqrt(np.sum(self.u**2, axis=-1)) * self.area
        self.H = np.abs(e) + 0.5 * np.sum(self.u**2, axis=-1) + self.p / self.rho


def wallfaces(mb):
    blk = mb.blocks[0]
    mb.apply_bcs("preadvflux")
    mb.launch(mb.kernels["advflux"](mb.regs[0]), "interior")
    mb.apply_bcs("postflux")
    return [WallFace(mb, i) for i in blk.intfaces if i.nbr is None]


@pytest.mark.parametrize("into", [False, True], ids=["along", "into"])
@pytest.mark.parametrize("keys, kind", list(cases()))
def test_only_momentum_crosses_a_slip_wall(backend, keys, kind, into):
    mb = walled(backend, kind, into, **keys)
    for face in wallfaces(mb):
        # the wall pressure: the cell's brought to rest, its speed toward the wall along the outward normal
        pw = np.maximum(face.p + face.rho * face.c * face.outward * face.un, 0.0)
        F, pS = face.F, pw * face.area
        # the mass, energy and species rows: nothing, exactly
        assert (F[..., 0] == 0.0).all()
        assert (F[..., 4] == 0.0).all()
        assert (F[..., 5:] == 0.0).all()
        # the momentum, over the wall pressure: normal to the wall
        force = F[..., 1:4] - pS[..., None] * face.n
        normal = np.sum(force * face.n, axis=-1)
        tangential = force - normal[..., None] * face.n
        assert (np.abs(tangential) <= 1e-14 * pS[..., None]).all()
        # the wall pressure alone: measured within 1.7e-18
        assert (np.abs(normal) <= 1e-14 * pS).all()
    # over a step the box keeps its mass, species and energy, and along the walls its momentum too: measured within 1.3e-15
    blk = mb.blocks[0]
    before = totals(mb)
    Q = blk.Q.get()[blk.interior]
    speed = np.sqrt(np.sum((Q[..., 1:4] / Q[..., :1]) ** 2, axis=-1)).max()
    mb.step(mb.cfg["simulation"]["dt"])
    change = np.abs(totals(mb) - before)
    mass = before[0]
    scale = np.array([mass, mass * speed, mass * speed, mass * speed, abs(before[4]), mass])
    rows = [0, 4, 5] if into else slice(None)
    assert (change[rows] <= 1e-14 * scale[rows]).all()
