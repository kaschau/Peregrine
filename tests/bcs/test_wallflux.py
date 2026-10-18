import itertools

import numpy as np
import pytest

import peregrine as pg

from tests.gases import configure, primitives
from tests.bcs.bcblock import PerturbedCube

# a wall writes its own face flux after the fluxes, so it passes nothing under any flux function, rusanov included
FLUXES = ("kepaec", "ausmup", "rusanov")
# every wall kind: whether the flow sticks to it, moves with it, and whether it holds a temperature
WALLS = {
    "slp-adia-wall": (False, False, False),
    "slp-isot-wall": (False, False, True),
    "no-slp-adia-wall": (True, False, False),
    "no-slp-isot-wall": (True, False, True),
    "mov-adia-wall": (True, True, False),
    "mov-isot-wall": (True, True, True),
}
# the wall's temperature, and the velocity of one that moves, along the walls
TWALL, UWALL = 380.0, (3.0, 0.0, -2.0)


def solver(backend, physics, kind, primary, recon="constant", gas="air", walled="between"):
    config = pg.Config()
    configure(config, gas, physics)
    config["adv-flux"]["primary"] = primary
    config["adv-flux"]["recon"] = recon
    if recon == "muscl":
        config["adv-flux"]["limiter"] = "koren"
    _, moving, isothermal = WALLS[kind]
    entry = {"kind": kind}
    if isothermal:
        entry["T"] = TWALL
    if moving:
        entry.update(zip("uvw", UWALL))
    config["bcvals"]["wall"] = entry
    box = dict(mbdims=[1, 1, 1], blkdims=[8, 7, 6], lengths=[1, 1, 1])
    if walled == "all":
        mesh = pg.mesher.CubeMesher(**box, defaultbc="wall")
    elif walled == "jostled":
        mesh = PerturbedCube(np.random.default_rng(3), **box, defaultbc="wall")
    else:
        mesh = pg.mesher.CubeMesher(**box, periodic=[True, False, True], bcnames={3: "wall", 4: "wall"})
    return pg.get_solver(config, mesh, backend(config))


def prims(mb, flow, T=None):
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    shape, ng = q.shape[:3], blk.ng
    i, j, k = np.meshgrid(*(np.arange(n, dtype=float) for n in shape), indexing="ij")
    x, y, z = ((c - ng + 0.5) / (n - 2 * ng) for c, n in zip((i, j, k), shape))
    q[..., 0] = 101325.0 * (1.0 + 0.05 * np.sin(np.pi * y) * np.cos(np.pi * x))
    q[..., 1:4] = 0.0
    q[..., 4] = 360.0 * (1.0 + 0.08 * np.cos(np.pi * y) + 0.04 * np.sin(np.pi * z)) if T is None else T
    if flow == "into":
        q[..., 1] = 20.0 * np.sin(np.pi * x) + 5.0
        q[..., 2] = 15.0 * np.cos(np.pi * y) - 3.0
        q[..., 3] = 10.0 * np.sin(np.pi * z) + 7.0
    elif flow:
        q[..., 1] = 40.0 * np.sin(np.pi * y) + 10.0
        q[..., 3] = 20.0 * np.cos(np.pi * y)
    if q.shape[-1] > 5:
        q[..., 5] = 0.3 + 0.1 * np.sin(np.pi * y)
    if q.shape[-1] > 6:
        q[..., 6] = 0.2 - 0.05 * np.cos(np.pi * z)
    return q


def fill(mb, flow):
    mb.set_prims([prims(mb, flow)])


def rhs(mb):
    stage = mb.integ.stageregs(mb.regs)[0]
    mb.rhs(Q=stage.Q, dQ=stage.dQ)
    return stage


def fluxes(blk):
    return {d: getattr(blk, f"{d}F").get().copy() for d in "ijk"}


def walls(blk):
    return [i for i in blk.intfaces if i.nbr is None]


def adjacent(interface, array):
    ng = interface.ng
    planes = np.moveaxis(array, interface.axis, 0)
    plane = planes[ng] if interface.is_low else planes[len(planes) - ng - 1]
    return plane[ng:-ng, ng:-ng]


def areas(blk, interface):
    return interface.faceplane(getattr(blk, f"{'ijk'[interface.axis]}S").get())


def scales(F):
    scale = np.abs(F).reshape(-1, F.shape[-1]).max(axis=0)
    # a species absent from the flow has a flux at the roundoff floor everywhere, so the species rows share the largest scale
    scale[5:] = scale[5:].max(initial=0.0)
    return scale


def assert_nothing_advected(mb, F, eps=1e-13):
    blk = mb.blocks[0]
    state = blk.state.get()
    p = list(mb.sim.statefields).index("p")
    for interface in walls(blk):
        d = "ijk"[interface.axis]
        Fw = interface.faceplane(F[d])
        for row in (0, 4, *range(5, Fw.shape[-1])):
            assert (Fw[..., row] == 0.0).all()
        want = adjacent(interface, state)[..., p, None] * areas(blk, interface)
        assert np.allclose(Fw[..., 1:4], want, rtol=0.0, atol=eps * np.abs(want).max())


def assert_totals_stand(mb, stage, F, eps=1e-12):
    blk = mb.blocks[0]
    dQ = getattr(blk, stage.dQ).get()[blk.interior]
    volume = 1.0 / blk.detJinv.get()[blk.interior]
    moved = np.abs((dQ * volume[..., None]).sum(axis=(0, 1, 2)))
    scale = np.max([np.abs(f).reshape(-1, f.shape[-1]).max(axis=0) for f in F.values()], axis=0)
    for row in (0, 4, *range(5, dQ.shape[-1])):
        assert moved[row] <= eps * scale[row]


def cases(kinds, recons, layouts, physics):
    out = []
    for kind, primary, recon, (walled, flow) in itertools.product(kinds, FLUXES, recons, layouts):
        out.append(pytest.param(kind, primary, recon, walled, flow, id=f"{kind}-{primary}-{recon}-{walled}-{'flow' if flow else 'rest'}"))
    return out


EULER = cases(("slp-adia-wall", "slp-isot-wall"), ("constant", "muscl"), (("all", False), ("between", True)), "euler")


@pytest.mark.parametrize("kind,primary,recon,walled,flow", EULER)
def test_an_euler_wall_passes_nothing(backend, kind, primary, recon, walled, flow):
    mb = solver(backend, "euler", kind, primary, recon, walled=walled)
    fill(mb, flow)
    stage = rhs(mb)
    F = fluxes(mb.blocks[0])
    assert_nothing_advected(mb, F)
    assert_totals_stand(mb, stage, F)


NAVSTOKES = cases(WALLS, ("constant",), (("between", True), ("between", False)), "navier-stokes")


# the coefficients at exactly T, through the solver's own kernels on primitives rather than the eos's inversion; the flow is refilled after
def wallcoefs(mb, flow, T):
    blk = mb.blocks[0]
    blk.replace("q", prims(mb, flow, T))
    mb._forget("q")
    mb.launch(mb.kernels["fromprims"](Q=mb.regs[0]), "all")
    mb.launch(mb.kernels["trans"](mb.regs[0]), "all")
    coefs = blk.trstate.get().copy()
    fill(mb, flow)
    return coefs


# the adjacent cell's pressure is the wall pressure only where nothing flows into the wall
def viscous(mb, interface, F):
    blk = mb.blocks[0]
    p = list(mb.sim.statefields).index("p")
    Fw = interface.faceplane(F["ijk"[interface.axis]]).copy()
    Fw[..., 1:4] -= adjacent(interface, blk.state.get())[..., p, None] * areas(blk, interface)
    return Fw


@pytest.mark.parametrize("kind,primary,recon,walled,flow", NAVSTOKES)
def test_a_viscous_wall_passes_no_mass_or_species(backend, kind, primary, recon, walled, flow):
    mb = solver(backend, "navier-stokes", kind, primary, recon, walled=walled)
    fill(mb, flow)
    stage = rhs(mb)
    blk = mb.blocks[0]
    F = fluxes(blk)
    for interface in walls(blk):
        Fw = interface.faceplane(F["ijk"[interface.axis]])
        for row in (0, *range(5, Fw.shape[-1])):
            assert (Fw[..., row] == 0.0).all()
    dQ = getattr(blk, stage.dQ).get()[blk.interior]
    volume = 1.0 / blk.detJinv.get()[blk.interior]
    moved = np.abs((dQ * volume[..., None]).sum(axis=(0, 1, 2)))
    scale = np.max([np.abs(f).reshape(-1, f.shape[-1]).max(axis=0) for f in F.values()], axis=0)
    for row in (0, *range(5, dQ.shape[-1])):
        assert moved[row] <= 1e-12 * scale[row]


def twopoint(mb, interface, coef, held, rows):
    blk = mb.blocks[0]
    axis = interface.axis
    xf = interface.faceplane(getattr(blk, f"{'ijk'[axis]}xc").get())[..., axis]
    dn = np.abs(xf - adjacent(interface, blk.xcc.get())[..., axis])
    sign = -1.0 if interface.is_low else 1.0
    S = areas(blk, interface)[..., axis]
    return -coef * sign * (held - rows) / dn * S


DIFFUSIVE = [
    pytest.param(kind, gas, flow, id=f"{kind}-{gas}-{'flow' if flow else 'rest'}")
    for kind, (gas, flow) in itertools.product(WALLS, (("air", True), ("air", False), ("CH4_O2", True)))
]


@pytest.mark.parametrize("kind,gas,flow", DIFFUSIVE)
def test_the_viscous_flux_at_a_wall(backend, kind, gas, flow):
    sticks, moving, isothermal = WALLS[kind]
    mb = solver(backend, "navier-stokes", kind, "kepaec", gas=gas)
    blk = mb.blocks[0]
    coefs = wallcoefs(mb, flow, TWALL) if isothermal else None
    fill(mb, flow)
    rhs(mb)
    coefs = blk.trstate.get() if coefs is None else coefs
    F = fluxes(blk)
    Q, state = blk.Q.get(), blk.state.get()
    T = list(mb.sim.statefields).index("T")
    eps = 1e-13
    for interface in walls(blk):
        axis = interface.axis
        Fw, scale = viscous(mb, interface, F), scales(F["ijk"[axis]])
        assert np.abs(Fw[..., 1 + axis]).max() <= eps * scale[1:4].max()
        cell = adjacent(interface, Q)
        velocity = cell[..., 1:4] / cell[..., :1]
        mu, kappa = (adjacent(interface, coefs)[..., c] for c in (0, 1))
        for c in (t for t in range(3) if t != axis):
            if sticks:
                want = twopoint(mb, interface, mu, UWALL[c] if moving else 0.0, velocity[..., c])
                assert np.allclose(Fw[..., 1 + c], want, rtol=eps, atol=eps * scale[1:4].max())
            else:
                assert np.abs(Fw[..., 1 + c]).max() <= eps * scale[1:4].max()
        if isothermal and not moving:
            want = twopoint(mb, interface, kappa, TWALL, adjacent(interface, state)[..., T])
            assert np.allclose(Fw[..., 4], want, rtol=eps, atol=eps * scale[4])
        elif not moving:
            assert np.abs(Fw[..., 4]).max() <= eps * scale[4]


@pytest.mark.parametrize("kind", ["mov-adia-wall", "mov-isot-wall"])
def test_a_moving_wall_works_on_the_fluid(backend, kind):
    _, _, isothermal = WALLS[kind]
    mb = solver(backend, "navier-stokes", kind, "kepaec")
    blk = mb.blocks[0]
    coefs = wallcoefs(mb, True, TWALL) if isothermal else None
    fill(mb, True)
    rhs(mb)
    coefs = blk.trstate.get() if coefs is None else coefs
    F = fluxes(blk)
    state = blk.state.get()
    T = list(mb.sim.statefields).index("T")
    for interface in walls(blk):
        Fw = viscous(mb, interface, F)
        want = Fw[..., 1:4] @ np.array(UWALL)
        if isothermal:
            kappa = adjacent(interface, coefs)[..., 1]
            want = want + twopoint(mb, interface, kappa, TWALL, adjacent(interface, state)[..., T])
        assert np.allclose(Fw[..., 4], want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


JOSTLED = [pytest.param(kind, gas, id=f"{kind}-{gas}") for kind, gas in (("slp-isot-wall", "air"), ("no-slp-adia-wall", "air"), ("mov-isot-wall", "CH4_O2"))]


@pytest.mark.parametrize("kind,gas", JOSTLED)
def test_the_wall_flux_on_a_jostled_face(backend, kind, gas):
    sticks, moving, isothermal = WALLS[kind]
    mb = solver(backend, "navier-stokes", kind, "kepaec", gas=gas, walled="jostled")
    blk = mb.blocks[0]
    coefs = wallcoefs(mb, "into", TWALL) if isothermal else None
    fill(mb, "into")
    rhs(mb)
    coefs = blk.trstate.get() if coefs is None else coefs
    F = fluxes(blk)
    Q, state = blk.Q.get(), blk.state.get()
    fields = list(mb.sim.statefields)
    p, T, c = (fields.index(f) for f in ("p", "T", "c"))
    wall = np.array(UWALL if moving else (0.0, 0.0, 0.0))
    for interface in walls(blk):
        axis = interface.axis
        outward = -1.0 if interface.is_low else 1.0
        S = areas(blk, interface)
        Sabs = np.linalg.norm(S, axis=-1)
        n = S / Sabs[..., None]
        xf = interface.faceplane(getattr(blk, f"{'ijk'[axis]}xc").get())
        cell, st = adjacent(interface, Q), adjacent(interface, state)
        dn = np.abs(np.sum((adjacent(interface, blk.xcc.get()) - xf) * n, axis=-1))
        rho, u = cell[..., 0], cell[..., 1:4] / cell[..., :1]
        un = outward * np.sum(u * S, axis=-1) / Sabs
        # the wall pressure carries the acoustic correction rho c un for flow into the wall
        pw = np.maximum(st[..., p] + rho * st[..., c] * un, 0.0)
        dun = outward * (wall - u) / dn[..., None]
        div = np.sum(dun * n, axis=-1, keepdims=True)
        dun = div * n + (1.0 if sticks else 0.0) * (dun - div * n)
        mu, kappa = (adjacent(interface, coefs)[..., i] for i in (0, 1))
        tS = -mu[..., None] * Sabs[..., None] * (dun + div * n / 3.0)
        dT = TWALL - st[..., T] if isothermal else 0.0
        energy = np.sum(wall * tS, axis=-1) - kappa * outward * dT / dn * Sabs
        Fw = interface.faceplane(F["ijk"[axis]])
        for row in (0, *range(5, Fw.shape[-1])):
            assert (Fw[..., row] == 0.0).all()
        pressure = pw[..., None] * S
        assert np.allclose(Fw[..., 1:4], pressure + tS, rtol=0.0, atol=1e-13 * np.abs(pressure).max())
        # the stress rides on the pressure, so it is held to the pressure's roundoff and its own
        assert np.allclose(Fw[..., 1:4] - pressure, tS, rtol=0.0, atol=1e-14 * np.abs(pressure).max() + 1e-12 * np.abs(tS).max())
        assert np.allclose(Fw[..., 4], energy, rtol=0.0, atol=1e-12 * np.abs(energy).max())
