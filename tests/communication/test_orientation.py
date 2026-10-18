import itertools

import numpy as np
import pytest

import peregrine as pg
from peregrine.partition import Conditioner

from tests.communication.test_xchg import configured, recons


# an orientation digit per axis of ours names the neighbour's axis it runs along: 1-3 for +i, +j, +k and 4-6 for -i, -j, -k
def signedperm(S):
    axes = tuple((int(d) - 1) % 3 for d in S)
    flips = tuple(int(d) > 3 for d in S)
    return axes, flips


def orientations():
    valid = []
    for axes in itertools.permutations((0, 1, 2)):
        for signs in itertools.product((1, -1), repeat=3):
            P = np.zeros((3, 3))
            for m, (a, s) in enumerate(zip(axes, signs)):
                P[m, a] = s
            # the other 24 signed permutations are reflections, which no right handed grid can produce
            if np.linalg.det(P) > 0:
                valid.append(
                    "".join(
                        str(a + 1 + (3 if s < 0 else 0)) for a, s in zip(axes, signs)
                    )
                )
    assert len(valid) == 24
    return valid


def reorient(a, S):
    axes, flips = signedperm(S)
    out = a
    for m, f in enumerate(flips):
        if f:
            out = np.flip(out, axis=m)
    return np.moveaxis(out, (0, 1, 2), axes)


def unorient(a, S):
    axes, flips = signedperm(S)
    out = np.moveaxis(a, axes, (0, 1, 2))
    for m, f in enumerate(flips):
        if f:
            out = np.flip(out, axis=m)
    return out


class ReorientedPair(pg.mesher.CubeMesher):
    def __init__(self, S, **kwargs):
        super().__init__(**kwargs)
        self.S = S

    def fill(self, mb):
        super().fill(mb)
        if self.S == "123":
            return
        axes, flips = signedperm(self.S)
        # the partitioner takes the inverse: which old axis each new one holds, and which new ones run backwards
        perm = [axes.index(m) for m in range(3)]
        newflips = [flips[perm[m]] for m in range(3)]
        Conditioner().reorient(mb, mb.get_blk(1), perm, newflips)


VARLIST = ["nodes", "Q"]


def traded(backend, S, recon, seed):
    mesh = ReorientedPair(S, mbdims=[2, 1, 1], blkdims=[6, 3, 2], lengths=[2, 1, 1], defaultbc="wall")
    config = configured(recon)
    mb = pg.get_solver(config, mesh, backend(config))
    assert mb.ng == recons[recon]

    # drawn in the reference layout, so block 1 holds the same physical field however it is stored
    rng = np.random.default_rng(seed)
    # both blocks have the reference shape in the reference layout
    for blk in mb.blocks:
        values = rng.random(mb.blocks[0].Q.shape)
        blk.Q.set(values if blk.nblki == 0 else reorient(values, S))

    for var in VARLIST:
        g = mb.backend.graph("test")
        g.add_send("sent", var)
        g.add_recv("landed", var)
        g.bind(mb)
        g.run()
    return mb


# the identity run is the ground truth, built once per depth
_refs = {}


def reference(backend, recon, seed):
    if (recon, seed) not in _refs:
        mb = traded(backend, "123", recon, seed)
        ng = mb.ng
        a, b = (blk.Q.get() for blk in mb.blocks)
        # a trade that wrote nothing would pass the comparison below, so the landing itself is checked first
        for g in range(ng):
            assert np.array_equal(a[-ng + g, ng:-ng, ng:-ng], b[ng + g, ng:-ng, ng:-ng])
        _refs[recon, seed] = (
            {v: getattr(mb.blocks[0], v).get() for v in VARLIST},
            {v: getattr(mb.blocks[1], v).get() for v in VARLIST},
        )
    return _refs[recon, seed]


@pytest.mark.parametrize("recon", list(recons))
@pytest.mark.parametrize("S", orientations())
def test_orientation(backend, S, recon):
    seed = 20260901
    ref0, ref1 = reference(backend, recon, seed)

    mb = traded(backend, S, recon, seed)
    for var in VARLIST:
        assert np.array_equal(getattr(mb.blocks[0], var).get(), ref0[var]), (S, var, "blk0")
        assert np.array_equal(getattr(mb.blocks[1], var).get(), reorient(ref1[var], S)), (S, var, "blk1")
