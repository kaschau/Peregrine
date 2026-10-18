import numpy as np
import pytest

from tests.gases import primitives
from tests.simulator import oracle as orc
from tests.simulator import support as sp

# every flux function on the two cells as they are, and the Riemann solvers on muscl's sides; kepaec is the scheme of the two cells as they are, so muscl is no composition for it
pairs = [("kepaec", "constant"), ("rusanov", "constant"), ("ausmup", "constant"), ("hllc", "constant"), ("rusanov", "muscl"), ("ausmup", "muscl"), ("hllc", "muscl")]
ids = ["-".join(pair) for pair in pairs]


def adv(primary, recon):
    return dict(primary=primary, recon=recon, limiter="koren" if recon == "muscl" else None)


def smooth(mb, seed=3):
    blk = mb.blocks[0]
    rng = np.random.default_rng(seed)
    X = blk.xcc.get()
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    ph = rng.uniform(0.0, 2.0 * np.pi, 8)
    q = primitives(mb, blk)
    q[..., 0] = 101325.0 * (1 + 0.2 * np.sin(2 * np.pi * x + ph[0]) * np.cos(2 * np.pi * y + ph[1]))
    q[..., 1] = 60.0 * np.cos(2 * np.pi * x + ph[2]) + 20.0 * np.sin(2 * np.pi * z + ph[3])
    q[..., 2] = 15.0 * np.sin(2 * np.pi * y + ph[4]) - 25.0
    q[..., 3] = -10.0 * np.cos(2 * np.pi * z + ph[5]) + 5.0 * np.sin(2 * np.pi * x)
    q[..., 4] = 300.0 * (1 + 0.3 * np.cos(2 * np.pi * x + ph[6]) * np.sin(2 * np.pi * z + ph[7]))
    q[..., 5] = 0.5 + 0.3 * np.sin(2 * np.pi * (x + y) + ph[0])
    mb.set_prims([q])


def sides(f, recon):
    if recon == "constant":
        return orc.cell(f.Q[-1], f.state[-1]), orc.cell(f.Q[0], f.state[0])
    return orc.muscl(sp.binary, f.Q, f.state)


def reach(recon):
    return 2 if recon == "muscl" else 1


def close(got, want, within):
    rows = want.reshape(-1, want.shape[-1])
    scale = np.abs(rows).max(axis=0)
    assert (np.abs(got - want) <= within * scale).all()


def fluxed(mb):
    mb.launch(mb.kernels["advflux"]("Q"), "interior")


@pytest.mark.parametrize("primary,recon", pairs, ids=ids)
def test_the_face_flux_is_its_formula(backend, primary, recon):
    mb = sp.box(backend, **adv(primary, recon))
    smooth(mb)
    fluxed(mb)
    for axis in range(3):
        f = sp.faces(mb, axis, reach(recon))
        L, R = sides(f, recon)
        # measured at most 4.3e-15 of a row's largest
        close(f.F[..., : mb.sim.ne], orc.functions[primary](L, R, f.S), 1e-13)


@pytest.mark.parametrize("primary,recon", pairs, ids=ids)
def test_the_apply_is_the_flux_divergence(backend, primary, recon):
    mb = sp.box(backend, **adv(primary, recon))
    smooth(mb)
    stage = sp.stepregs(mb)
    mb.rhs(Q=stage.Q, dQ=stage.dQ)
    blk = mb.blocks[0]
    ng = blk.ng
    inner = sp.interior(blk)
    iF, jF, kF = blk.iF.get(), blk.jF.get(), blk.kF.get()
    low = (iF[ng : -ng - 1, ng:-ng, ng:-ng], jF[ng:-ng, ng : -ng - 1, ng:-ng], kF[ng:-ng, ng:-ng, ng : -ng - 1])
    high = (iF[ng + 1 : -ng, ng:-ng, ng:-ng], jF[ng:-ng, ng + 1 : -ng, ng:-ng], kF[ng:-ng, ng:-ng, ng + 1 : -ng])
    # summed as the kernel sums, the low faces in then the high faces out, so the match is bitwise
    want = (low[0] + low[1] + low[2] - high[0] - high[1] - high[2]) * blk.detJinv.get()[inner][..., None]
    assert np.array_equal(getattr(blk, stage.dQ).get()[inner], want)


def blended(backend, values):
    return sp.box(backend, formulation="blended", primary="kepaec", recon="constant", secondary="rusanov", switch="jameson", values=values)


def slab(mb, column, low, high):
    smooth(mb)
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    q[..., column] = np.where(blk.xcc.get()[..., 0] < 0.5, low, high)
    mb.set_prims([q])


def test_the_blend_is_weighed_by_the_switch(backend):
    # the smooth state: every weight below one, the primary carried through where it is small
    values = {"p-gain": 5.0, "gamma-gain": 1.0}
    mb = blended(backend, values)
    smooth(mb)
    fluxed(mb)
    largest = 0.0
    for axis in range(3):
        f = sp.faces(mb, axis, 2)
        L, R = sides(f, "constant")
        shock, contact = orc.jameson(f.state, **{"pgain": values["p-gain"], "ggain": values["gamma-gain"]})
        w = np.maximum(shock, contact)
        largest = max(largest, w.max())
        assert w.min() < 0.01
        # measured 4.4e-16
        close(f.F[..., : mb.sim.ne], (1 - w)[..., None] * orc.kepaec(L, R, f.S) + w[..., None] * orc.rusanov(L, R, f.S), 1e-13)
    assert largest < 1.0


@pytest.mark.parametrize(
    "column,low,high,values",
    [
        # a pressure jump of ten: the pressure sensor past one, clipped, so both sensors are one
        (0, 101325.0, 10 * 101325.0, {"p-gain": 5.0, "gamma-gain": 1.0}),
        # a composition jump at one pressure: the gamma sensor alone, past one at this gain
        (5, 0.1, 0.9, {"p-gain": 5.0, "gamma-gain": 1000.0}),
    ],
)
def test_the_switch_clips_at_one(backend, column, low, high, values):
    mb = blended(backend, values)
    slab(mb, column, low, high)
    fluxed(mb)
    f = sp.faces(mb, 0, 2)
    L, R = sides(f, "constant")
    shock, contact = orc.jameson(f.state, pgain=values["p-gain"], ggain=values["gamma-gain"])
    w = np.maximum(shock, contact)
    assert w.max() == 1.0
    # the composition jump moves the contact sensor alone
    assert (shock.max() == 1.0) == (column == 0)
    # measured 3.9e-15
    close(f.F[..., : mb.sim.ne], (1 - w)[..., None] * orc.kepaec(L, R, f.S) + w[..., None] * orc.rusanov(L, R, f.S), 1e-13)


# near Mach 0.1, where the cutoff Mach sets AUSM+-up's scaling
def slow(mb):
    blk = mb.blocks[0]
    X = blk.xcc.get()
    q = primitives(mb, blk)
    q[..., 0] = 101325.0 * (1 + 0.01 * np.sin(2 * np.pi * X[..., 0]))
    q[..., 1] = 35.0 + 5.0 * np.cos(2 * np.pi * X[..., 1])
    q[..., 2] = 3.0 * np.sin(2 * np.pi * X[..., 2])
    q[..., 3] = -2.0
    q[..., 4] = 300.0 * (1 + 0.05 * np.cos(2 * np.pi * X[..., 2]))
    q[..., 5] = 0.6
    mb.set_prims([q])


def test_the_cutoff_mach_is_the_sections_value(backend):
    Mco = 0.05
    mb = sp.box(backend, primary="ausmup", recon="constant", values={"cutoff-mach": Mco})
    slow(mb)
    fluxed(mb)
    assert all("Mco = 0.05" in k.text for k in mb.kernels["advflux"]("Q").kernels)
    for axis in range(3):
        f = sp.faces(mb, axis, 1)
        L, R = sides(f, "constant")
        assert (np.abs(L["u"]) / L["c"]).max() < 0.2
        want = orc.ausmup(L, R, f.S, Mco=Mco)
        close(f.F[..., : mb.sim.ne], want, 1e-13)
        # the value reached the kernel: at the default cutoff the pressure diffusion is another (measured 6.6e-2)
        default = orc.ausmup(L, R, f.S)
        assert np.abs(want - default).max() > 1e-3 * np.abs(want).max()


def test_rusanov_takes_each_sides_own_sound_speed(backend):
    # a hot fast state against a cold slow one: Davis's max(|uL| + cL, |uR| + cR) is a fifth above the pairing of each side's speed with the other's sound speed, so the oracle tells them apart
    mb = sp.line(backend, 41, gas=sp.air, primary="rusanov", recon="constant")
    blk = mb.blocks[0]
    R, g = sp.gasconstant(sp.air), sp.gamma(sp.air)
    cL, cR = np.sqrt(g * R * 1200.0), np.sqrt(g * R * 300.0)
    hot = blk.xcc.get()[..., 0] < 0.5
    q = primitives(mb, blk)
    q[..., 0] = 101325.0
    q[..., 1] = np.where(hot, 1.5 * cL, 1.0 * cR)
    q[..., 4] = np.where(hot, 1200.0, 300.0)
    mb.set_prims([q])
    fluxed(mb)
    f = sp.faces(mb, 0, 1)
    L, Rr = sides(f, "constant")
    close(f.F[..., : mb.sim.ne], orc.rusanov(L, Rr, f.S), 1e-13)
    area = np.sqrt(orc.dot(f.S, f.S))
    unL, unR = orc.dot(orc.velocity(L), f.S) / area, orc.dot(orc.velocity(Rr), f.S) / area
    davis = np.maximum(np.abs(unL) + L["c"], np.abs(unR) + Rr["c"])
    crossed = np.maximum(np.abs(unL) + Rr["c"], np.abs(unR) + L["c"])
    assert (crossed / davis).min() < 0.81


@pytest.mark.parametrize("name", ["kepaec", "rusanov", "ausmup", "hllc", "muscl-koren-rusanov", "muscl-koren-ausmup", "muscl-koren-hllc"])
def test_the_flux_is_skew_symmetric_under_a_mirror(backend, name):
    # the line reversed cell for cell with u negated: face f becomes face N - f, and its flux's mass, cross momentum, energy and species rows negate while the normal momentum keeps
    mb = sp.line(backend, 41, gas=sp.binary, **sp.schemes[name])
    blk = mb.blocks[0]
    x = blk.xcc.get()[..., 0]
    rng = np.random.default_rng(5)
    ph = rng.uniform(0.0, 2.0 * np.pi, 6)
    q = primitives(mb, blk)
    q[..., 0] = 101325.0 * (1 + 0.2 * np.sin(2 * np.pi * x + ph[0]))
    q[..., 1] = 60.0 * np.cos(2 * np.pi * x + ph[1]) + 20.0
    q[..., 2] = 15.0 * np.sin(4 * np.pi * x + ph[2])
    q[..., 3] = -10.0 * np.cos(2 * np.pi * x + ph[3])
    q[..., 4] = 300.0 * (1 + 0.3 * np.cos(2 * np.pi * x + ph[4]))
    q[..., 5] = 0.5 + 0.3 * np.sin(2 * np.pi * x + ph[5])
    mb.set_prims([q])
    fluxed(mb)
    F = sp.faces(mb, 0, 1).F[:, 0, 0, : mb.sim.ne]
    mirrored = q[::-1].copy()
    mirrored[..., 1] *= -1.0
    mb.set_prims([mirrored])
    fluxed(mb)
    Fm = sp.faces(mb, 0, 1).F[:, 0, 0, : mb.sim.ne]
    signs = np.array([-1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    scale = np.abs(F).max(axis=-1, keepdims=True)
    # measured at most 1e-14 of a face's largest row
    assert (np.abs(Fm[::-1] - signs * F) <= 1e-13 * scale).all()
