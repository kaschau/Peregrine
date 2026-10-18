import numpy as np
import pytest

import peregrine as pg

from tests.navstokes import support

Tw, Uw, L = 280.0, 50.0, 1e-2

# every kind with what its entry gives; a moving wall moves along x on the j faces alone, its kind still on the others
WALLS = {
    "slp-adia-wall": {},
    "slp-isot-wall": {"T": Tw},
    "no-slp-adia-wall": {},
    "mov-adia-wall": {"u": Uw, "v": 0.0, "w": 0.0},
    "no-slp-isot-wall": {"T": Tw},
    "mov-isot-wall": {"u": Uw, "v": 0.0, "w": 0.0, "T": Tw},
}
ISOTHERMAL = {k for k in WALLS if "isot" in k}
MOVING = {k for k in WALLS if k.startswith("mov")}


def solver(backend, kind):
    config = support.config(support.mixed)
    values = WALLS[kind]
    config["bcvals"]["moving"] = {"kind": kind, **values}
    config["bcvals"]["still"] = {"kind": kind, **{k: (0.0 if k in "uvw" else v) for k, v in values.items()}}
    names = {1: "still", 2: "still", 3: "moving", 4: "moving", 5: "still", 6: "still"}
    mb = pg.get_solver(config, support.box([7, 6, 5], [L, L, L], bcnames=names), backend(config))
    blk = mb.blocks[0]
    x, y, z = (np.pi * blk.xcc.get()[..., c] / L for c in range(3))
    u = np.stack([10.0 * np.sin(x) * np.cos(y), 5.0 * np.cos(z) * np.sin(x), -3.0 * np.sin(y) * np.cos(z)], axis=-1)
    Y = np.stack([0.5 + 0.1 * np.cos(x) * np.sin(z), 0.25 + 0.1 * np.sin(x) * np.cos(y)], axis=-1)
    support.setprims(mb, blk, 101325.0 * (1.0 + 0.02 * np.sin(x) * np.cos(y)), u, 310.0 + 10.0 * np.sin(y) * np.cos(z), Y)
    return mb


class WallRows:
    def __init__(self, mb):
        self.mb = mb
        mb.rhs(Q="Q", dQ=mb.regs[-1])
        blk = mb.blocks[0]
        self.ng = ng = blk.ng
        self.Q, self.tr = blk.Q.get(), blk.trstate.get()
        self.T, self.p = (support.statefield(mb, blk, n) for n in ("T", "p"))
        self.xcc = blk.xcc.get()
        self.faces = {}
        for axis in range(3):
            a = "ijk"[axis]
            F, S = getattr(blk, f"{a}F").get(), getattr(blk, f"{a}S").get()
            # the size of every row over the axis's interior faces, which roundoff is measured against
            inner = [slice(ng, -ng)] * 3
            inner[axis] = slice(ng, F.shape[axis] - ng)
            scale = np.abs(F[tuple(inner)]).max(axis=(0, 1, 2))
            for f, c, sign in support.wallfaces(blk, axis):
                self.faces[axis, sign] = dict(
                    F=support.interior_plane(F, axis, f, ng),
                    S=support.interior_plane(S, axis, f, ng),
                    scale=scale,
                    p=support.interior_plane(self.p, axis, c, ng),
                    heat=self.heat(axis, f, c, sign, support.interior_plane(S, axis, f, ng)),
                )

    # at the cell's conductivity, which the constant model holds at the wall's temperature too
    def heat(self, axis, f, c, sign, S):
        ng = self.ng
        a = "ijk"[axis]
        kappa = support.interior_plane(self.tr, axis, c, ng)[..., 1]
        Tc = support.interior_plane(self.T, axis, c, ng)
        xf = support.interior_plane(getattr(self.mb.blocks[0], f"{a}xc").get(), axis, f, ng)[..., axis]
        dn = np.abs(xf - support.interior_plane(self.xcc, axis, c, ng)[..., axis])
        return -kappa * sign * (Tw - Tc) / dn * np.sqrt((S**2).sum(axis=-1))


@pytest.fixture(scope="module")
def cases():
    return {}


def rows(backend, cases, kind):
    if kind not in cases:
        cases[kind] = WallRows(solver(backend, kind))
    return cases[kind]


def moving(axis, kind):
    return axis == 1 and kind in MOVING


@pytest.mark.parametrize("kind", list(WALLS))
def test_no_mass_or_species_crosses_a_wall(backend, cases, kind):
    for face in rows(backend, cases, kind).faces.values():
        F, scale = face["F"], face["scale"]
        assert np.abs(F[..., 0]).max() <= 1e-14 * scale[0]
        assert np.abs(F[..., 5:]).max() <= 1e-13 * scale[5:].max()


@pytest.mark.parametrize("kind", list(WALLS))
def test_no_energy_crosses_an_adiabatic_wall_at_rest(backend, cases, kind):
    walls = rows(backend, cases, kind)
    for (axis, sign), face in walls.faces.items():
        if kind in ISOTHERMAL or moving(axis, kind):
            continue
        assert np.abs(face["F"][..., 4]).max() <= 1e-13 * face["scale"][4]


@pytest.mark.parametrize("kind", sorted(ISOTHERMAL))
def test_an_isothermal_wall_conducts_the_two_point_gradient(backend, cases, kind):
    walls = rows(backend, cases, kind)
    for (axis, sign), face in walls.faces.items():
        if moving(axis, kind):
            continue
        assert face["F"][..., 4] == pytest.approx(face["heat"], rel=1e-12)
        assert (sign * face["F"][..., 4] > 0.0).all()


@pytest.mark.parametrize("kind", sorted(MOVING))
def test_a_moving_wall_works_at_its_velocity(backend, cases, kind):
    walls = rows(backend, cases, kind)
    for sign in (-1.0, 1.0):
        face = walls.faces[1, sign]
        stress = face["F"][..., 1:4] - face["p"][..., None] * face["S"]
        want = Uw * stress[..., 0] + (face["heat"] if kind in ISOTHERMAL else 0.0)
        assert face["F"][..., 4] == pytest.approx(want, rel=1e-12)


def steps(mb, n):
    blk = mb.blocks[0]
    c = support.statefield(mb, blk, "c")[blk.interior].max()
    dt = 0.3 * (L / 6) / c
    before = support.totals(mb)
    for _ in range(n):
        mb.step(dt)
    return before, support.totals(mb)


@pytest.mark.parametrize("kind", list(WALLS))
def test_a_walled_box_keeps_its_mass_and_species(backend, cases, kind):
    before, after = steps(rows(backend, cases, kind).mb, 10)
    kept = [0, 5, 6]
    assert np.abs(after[kept] / before[kept] - 1.0).max() < 1e-13


@pytest.mark.parametrize("kind", ["slp-adia-wall", "no-slp-adia-wall", "slp-isot-wall", "no-slp-isot-wall"])
def test_a_still_walled_box_keeps_or_loses_energy_as_its_walls_say(backend, cases, kind):
    before, after = steps(rows(backend, cases, kind).mb, 10)
    if kind in ISOTHERMAL:
        assert after[4] < before[4] * (1.0 - 1e-10)
    else:
        assert after[4] == pytest.approx(before[4], rel=1e-13)


def test_a_wall_moving_faster_than_the_fluid_adds_energy(backend, cases):
    before, after = steps(rows(backend, cases, "mov-adia-wall").mb, 10)
    assert after[4] > before[4] * (1.0 + 1e-10)
