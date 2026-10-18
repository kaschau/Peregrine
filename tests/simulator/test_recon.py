import numpy as np
import pytest

import peregrine as pg
from peregrine.launch.kernel import Kernel
from peregrine.launch.ranges import FaceRange

from tests.gases import primitives
from tests.simulator import oracle as orc
from tests.simulator import support as sp

# the limiters under recon/, each with the oracle's numpy phi
limiters = orc.limiters
names = ("p", "T", "rho", "e", "c", "u", "v", "w")


class WithSides(pg.multiblock.Solver):
    def _decl_arrays(self):
        super()._decl_arrays()
        self.decl_array("isides", FaceRange, comps=2 * (8 + self.sim.mixture.ns - 1), axis=0)

    def _decl_kerns(self):
        super()._decl_kerns()
        section = self.cfg["adv-flux"]
        tpl = f"{__package__}.kernels.sides"
        self.kernels["sides"] = lambda Q: Kernel(self.provider, tpl, axis=0, options=dict(section), Q=Q, state="state", S="iS", out="isides")


def sides(mb):
    blk = mb.blocks[0]
    ng = blk.ng
    mb.launch(mb.kernels["sides"]("Q"), "interior")
    out = blk.isides.get()[:, ng, ng]
    n = blk.Q.shape[0] - 2 * ng
    out = out[ng : ng + n + 1]
    width = out.shape[-1] // 2
    both = []
    for o in (0, width):
        side = {k: out[:, o + i] for i, k in enumerate(names)}
        side["Y"] = out[:, o + 8 : o + width]
        both.append(side)
    return both


def line(backend, recon, limiter=None):
    return sp.line(backend, 41, periodic=False, gas=sp.binary, cls=WithSides, primary="rusanov", recon=recon, limiter=limiter)


def linear(mb):
    blk = mb.blocks[0]
    x = blk.xcc.get()[..., 0]
    q = primitives(mb, blk)
    for c, (a, b) in enumerate(((101325.0, 10132.5), (10.0, 50.0), (-5.0, 20.0), (3.0, -8.0), (300.0, 100.0), (0.2, 0.5))):
        q[..., c] = a + b * x
    mb.set_prims([q])
    return lambda x: np.stack([a + b * x for a, b in ((101325.0, 10132.5), (300.0, 100.0), (10.0, 50.0), (-5.0, 20.0), (3.0, -8.0), (0.2, 0.5))], axis=-1)


def rough(mb):
    blk = mb.blocks[0]
    rng = np.random.default_rng(5)
    q = primitives(mb, blk)
    shape = q.shape[:3]
    q[..., 0] = rng.uniform(0.5e5, 2e5, shape)
    q[..., 1:4] = rng.uniform(-100, 100, shape + (3,))
    q[..., 4] = rng.uniform(250, 900, shape)
    q[..., 5] = rng.uniform(0.0, 1.0, shape)
    mb.set_prims([q])


# the halo behind a wall is no part of the field
def inner(mb, reach):
    n = mb.blocks[0].Q.shape[0] - 2 * mb.ng
    return slice(reach, n + 1 - reach)


@pytest.mark.parametrize("limiter", limiters)
def test_a_linear_field_is_reconstructed_exactly(backend, limiter):
    # the slope ratio is one, phi is one, and each side lands on the face
    mb = line(backend, "muscl", limiter)
    at = linear(mb)
    xf = mb.blocks[0].ixc.get()[mb.ng:-mb.ng, mb.ng, mb.ng, 0]
    ok = inner(mb, 2)
    want = at(xf)[ok]
    scale = np.abs(want).max(axis=0)
    for side in sides(mb):
        got = np.stack([side[k] for k in ("p", "T", "u", "v", "w")] + [side["Y"][:, 0]], axis=-1)[ok]
        # measured 5e-16
        assert (np.abs(got - want) <= 1e-14 * scale).all()


@pytest.mark.parametrize("limiter", limiters)
def test_each_side_stays_within_its_two_cells(backend, limiter):
    mb = line(backend, "muscl", limiter)
    rough(mb)
    ok = inner(mb, 2)
    f = sp.faces(mb, 0, reach=1)
    L, R = orc.cell(f.Q[-1], f.state[-1]), orc.cell(f.Q[0], f.state[0])
    for side in sides(mb):
        for k in ("p", "T", "u", "v", "w"):
            lo, hi = np.minimum(L[k], R[k])[ok][:, 0, 0], np.maximum(L[k], R[k])[ok][:, 0, 0]
            got = side[k][ok]
            # measured no excursion at all
            assert (got >= lo - 1e-14 * (hi - lo)).all() and (got <= hi + 1e-14 * (hi - lo)).all()
        lo, hi = np.minimum(L["Y"], R["Y"])[ok][:, 0, 0], np.maximum(L["Y"], R["Y"])[ok][:, 0, 0]
        assert (side["Y"][ok] >= lo - 1e-14).all() and (side["Y"][ok] <= hi + 1e-14).all()


@pytest.mark.parametrize("limiter", limiters)
def test_the_sides_follow_sweby_with_the_limiters_phi(backend, limiter):
    # vL + phi / 2 (vR - vL) on the left of the face and vR - phi / 2 (vR - vL) on the right, phi of the ratio of the cell's upwind difference to the one across the face
    mb = line(backend, "muscl", limiter)
    rough(mb)
    ok = inner(mb, 2)
    f = sp.faces(mb, 0, reach=2)
    for got, want in zip(sides(mb), orc.muscl(sp.binary, f.Q, f.state, limiters[limiter])):
        for k in names:
            # measured 3.5e-16
            assert np.allclose(got[k][ok], want[k][ok][:, 0, 0], rtol=1e-13, atol=0.0), k
        assert np.allclose(got["Y"][ok], want["Y"][ok][:, 0, 0], rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("limiter", limiters)
def test_each_side_is_a_state_of_the_eos(backend, limiter):
    # whatever the limiters did to p, T and Y, rho, e and c are the eos's of them
    mb = line(backend, "muscl", limiter)
    rough(mb)
    ok = inner(mb, 2)
    for side in sides(mb):
        closed = orc.cpg(sp.binary, side["p"], side["T"], side["Y"])
        for k in ("rho", "c"):
            # measured 3.5e-16
            assert np.allclose(side[k][ok], closed[k][ok], rtol=1e-14, atol=0.0), k
        # e crosses zero on the line, so it is held at the row's scale: measured 6e-16
        got, want = side["e"][ok], closed["e"][ok]
        assert (np.abs(got - want) <= 1e-14 * np.abs(want).max()).all()


def test_constant_sides_are_the_two_cells(backend):
    mb = line(backend, "constant")
    rough(mb)
    f = sp.faces(mb, 0, reach=1)
    cells = (orc.cell(f.Q[-1], f.state[-1]), orc.cell(f.Q[0], f.state[0]))
    for got, want in zip(sides(mb), cells):
        for k in names:
            assert np.array_equal(got[k], want[k][:, 0, 0]), k
        assert np.array_equal(got["Y"], want["Y"][:, 0, 0])
