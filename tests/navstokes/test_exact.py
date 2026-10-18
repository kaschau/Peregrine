import numpy as np
import pytest

import peregrine as pg
from peregrine.simulator.mixture import Ru
from peregrine.simulator.mixture.polyfit import PolyFitMixin

from tests.navstokes import support


# exact only when x samples whole periods
def amplitude(f, x, k):
    return 2.0 * np.mean(f * np.sin(k * x))


# the alpha damping reduces to the three-point Laplacian on a uniform grid, so a wave decays at this rate rather than k^2
def kmod2(k, dx):
    return (2.0 - 2.0 * np.cos(k * dx)) / dx**2


def alongx(mb, blk):
    ng = blk.ng
    row = np.s_[ng:-ng, ng, ng]
    T, p = (support.statefield(mb, blk, n)[row] for n in ("T", "p"))
    return blk.xcc.get()[row][..., 0], blk.Q.get()[row], T, p


@pytest.mark.parametrize("nx", [20, 40])
def test_shear_and_scalar_decay_at_the_semi_discrete_rate(backend, nx):
    L = 2e-4
    config = support.config(support.alike)
    mb = pg.get_solver(config, support.line(nx, L), backend(config))
    blk = mb.blocks[0]
    ng = blk.ng
    x = blk.xcc.get()[..., 0]
    k, dx = 2.0 * np.pi / L, L / nx
    u = np.zeros(x.shape + (3,))
    u[..., 2] = np.sin(k * x)
    support.setprims(mb, blk, 101325.0, u, 300.0, (0.5 + 0.1 * np.sin(k * x))[..., None])
    sp = support.alike["A"]
    rho = blk.Q.get()[ng, ng, ng, 0]
    nu, D = sp["mu0"] / rho, sp["kappa0"] / (rho * sp["cp0"])
    cv = sp["cp0"] - Ru / sp["MW"]
    c = support.statefield(mb, blk, "c")[blk.interior].max()
    dt = 0.4 * dx / c
    n = int(round(0.5 / (D * k * k * dt)))
    vol = support.volumes(blk)

    def energies():
        Q = blk.Q.get()[blk.interior]
        T = support.statefield(mb, blk, "T")[blk.interior]
        kinetic = (0.5 * (Q[..., 1:4] ** 2).sum(axis=-1) / Q[..., 0] * vol).sum()
        return kinetic, (cv * Q[..., 0] * T * vol).sum()

    kinetic0, thermal0 = energies()
    for _ in range(n):
        mb.step(dt)
    t = n * dt
    xs, Q, T, p = alongx(mb, blk)
    Y, w = Q[:, 5] / Q[:, 0], Q[:, 3] / Q[:, 0]
    assert amplitude(Y, xs, k) / (0.1 * np.exp(-D * kmod2(k, dx) * t)) == pytest.approx(1.0, abs=1e-6)
    assert amplitude(w, xs, k) / np.exp(-nu * kmod2(k, dx) * t) == pytest.approx(1.0, abs=1e-6)
    assert np.ptp(T) < 1e-3 and np.ptp(p) < 1e-3
    # the kinetic energy the shear lost is in the temperature: the box is periodic
    kinetic1, thermal1 = energies()
    assert thermal1 - thermal0 == pytest.approx(kinetic0 - kinetic1, rel=1e-6)


# two constant-cp species of one weight differing in heat capacity and formation enthalpy, and the library's CO and N2 on the thermally perfect fits given one weight (their own differ by 1e-4, which stirs a Stefan flow and 2e-6 K)
air = {"mu0": 1.8e-5, "kappa0": 0.026}
layers = {
    "cpg": {
        "A": {"MW": 28.0, "cp0": 1000.0, "dHf298": 0.0, **air},
        "B": {"MW": 28.0, "cp0": 1300.0, "dHf298": 1.0e7, **air},
    },
    "tpg": {"CO": {"MW": 28.014, **air}, "N2": {"MW": 28.014, **air}},
}


# the temperature holds only with the species flux's enthalpy on the eos's own energy reference
@pytest.mark.parametrize("eos", list(layers))
def test_a_layer_diffuses_at_uniform_temperature(backend, eos):
    config = support.config(layers[eos], eos=eos)
    config["mixture"]["T-range"] = (200.0, 1000.0)
    nx, L = 40, 2e-4
    mb = pg.get_solver(config, support.line(nx, L), backend(config))
    blk = mb.blocks[0]
    x = blk.xcc.get()[..., 0]
    k, dx = 2.0 * np.pi / L, L / nx
    support.setprims(mb, blk, 101325.0, 0.0, 300.0, (0.5 + 0.1 * np.sin(k * x))[..., None])
    c = support.statefield(mb, blk, "c")[blk.interior].max()
    dt = 0.4 * dx / c
    before = support.totals(mb)
    xs, Q, _, _ = alongx(mb, blk)
    amp0 = amplitude(Q[:, 5] / Q[:, 0], xs, k)
    n = 1500
    for _ in range(n):
        mb.step(dt)
    xs, Q, T, p = alongx(mb, blk)
    u = Q[:, 1:4] / Q[:, :1]
    assert np.ptp(T) < 1e-9
    assert np.ptp(p) < 1e-6 and np.abs(u).max() < 1e-9
    # the layer decayed between the rates of the two compositions it spans, at Le = 1
    sp = [mb.sim.mixture.species[n] for n in mb.sim.mixture.spnames]
    cp = np.array([PolyFitMixin.evaluate(s["cppoly"], np.log(300.0)) * Ru / s["MW"] for s in sp])
    rho = Q[:, 0].mean()
    rates = [sp[0]["kappa0"] / (rho * (YA * cp[0] + (1.0 - YA) * cp[1])) for YA in (0.4, 0.6)]
    fallen = amplitude(Q[:, 5] / Q[:, 0], xs, k) / amp0
    lo, hi = sorted(np.exp(-D * kmod2(k, dx) * n * dt) for D in rates)
    assert lo * (1.0 - 1e-3) < fallen < hi * (1.0 + 1e-3)
    after = support.totals(mb)
    kept = [0, 4, 5]
    assert np.abs(after[kept] / before[kept] - 1.0).max() < 1e-12
    assert np.abs(after[1:4] - before[1:4]).max() < 1e-12 * before[0] * c


###############################################################################
# Couette flow: a still wall below, a wall moving at U above, periodic along x and z
###############################################################################
h, U, mu0, Pr, Tw, cp = 1e-2, 1.0, 1e-2, 0.7, 300.0, 1000.0


def couette(backend, ny, isothermal):
    species = {"Air": {"MW": 28.97, "cp0": cp, "mu0": mu0, "kappa0": mu0 * cp / Pr}}
    config = support.config(species)
    moving = {"u": U, "v": 0.0, "w": 0.0}
    if isothermal:
        config["bcvals"]["still"] = {"kind": "no-slp-isot-wall", "T": Tw}
        config["bcvals"]["moving"] = {"kind": "mov-isot-wall", "T": Tw, **moving}
    else:
        config["bcvals"]["still"] = {"kind": "no-slp-adia-wall"}
        config["bcvals"]["moving"] = {"kind": "mov-adia-wall", **moving}
    dy = h / ny
    mesh = support.box([2, ny + 1, 2], [dy, h, dy], periodic=[True, False, True], bcnames={3: "still", 4: "moving"})
    mb = pg.get_solver(config, mesh, backend(config))
    blk = mb.blocks[0]
    y = blk.xcc.get()[..., 1]
    u = np.zeros(y.shape + (3,))
    # the exact linear profile, so there is no transient and the dissipation is uniform in time
    u[..., 0] = U * y / h
    support.setprims(mb, blk, 101325.0, u, Tw, None)
    return mb


class Couette:
    def __init__(self, backend, ny, isothermal, taus):
        self.ny = ny
        self.mb = mb = couette(backend, ny, isothermal)
        self.blk = blk = mb.blocks[0]
        ng = blk.ng
        self.rho = blk.Q.get()[ng, ng, ng, 0]
        self.nu = mu0 / self.rho
        c = support.statefield(mb, blk, "c")[blk.interior].max()
        dy = h / ny
        dt = 0.4 * dy / c
        n = int(round(taus * h * h / self.nu / dt))
        for _ in range(n):
            mb.step(dt)
        self.t = n * dt
        column = np.s_[ng, ng:-ng, ng]
        self.y = blk.xcc.get()[column][..., 1]
        Q = blk.Q.get()[column]
        self.u, self.v = Q[:, 1] / Q[:, 0], Q[:, 2] / Q[:, 0]
        self.T, self.p = (support.statefield(mb, blk, name)[column] for name in ("T", "p"))
        mb.rhs(Q="Q", dQ=mb.regs[-1])
        jF = blk.jF.get()[ng, :, ng]
        area = np.sqrt((blk.jS.get()[ng, :, ng] ** 2).sum(axis=-1))
        (lo, _, _), (hi, _, _) = support.wallfaces(blk, 1)
        self.walls = {"still": (jF[lo], area[lo]), "moving": (jF[hi], area[hi])}


@pytest.fixture(scope="module")
def adiabatic(backend):
    return Couette(backend, 20, isothermal=False, taus=0.5)


@pytest.fixture(scope="module", params=[20, 40])
def isothermal(backend, request):
    return Couette(backend, request.param, isothermal=True, taus=2.0)


def test_adiabatic_couette_keeps_the_linear_profile(adiabatic):
    c = adiabatic
    assert np.abs(c.u - U * c.y / h).max() < 1e-7 * U
    assert np.abs(c.v).max() < 1e-6 * U
    assert np.ptp(c.T) < 1e-9


def test_adiabatic_couette_wall_rows(adiabatic):
    c = adiabatic
    for name, (F, area) in c.walls.items():
        assert F[1] == pytest.approx(-mu0 * U / h * area, rel=1e-7), name
        assert F[0] == 0.0, name
    still, area = c.walls["still"]
    assert abs(still[4]) < 1e-20
    assert still[2] == pytest.approx(c.p[0] * area, rel=1e-10)
    moving, area = c.walls["moving"]
    assert moving[2] == pytest.approx(c.p[-1] * area, rel=1e-10)


def test_adiabatic_couette_heats_by_the_dissipation(adiabatic):
    c = adiabatic
    cv = cp - Ru / 28.97
    assert c.T.mean() - Tw == pytest.approx(mu0 * (U / h) ** 2 * c.t / (c.rho * cv), rel=1e-6)


# energy enters, so in the + axis convention the row at the high wall is negative
def test_moving_adiabatic_wall_works_on_the_fluid(adiabatic):
    moving, area = adiabatic.walls["moving"]
    assert moving[4] == pytest.approx(-mu0 * U * U / h * area, rel=1e-7)


def test_isothermal_couette_keeps_the_linear_profile(isothermal):
    c = isothermal
    assert np.abs(c.u - U * c.y / h).max() < 1e-7 * U
    assert np.abs(c.v).max() < 1e-6 * U


def test_isothermal_couette_temperature_is_the_quadratic(isothermal):
    c = isothermal
    peak = Pr * U * U / (8.0 * cp)
    # the wall's two-point gradient to its temperature leaves this uniform offset
    offset = Pr * U * U / (2.0 * cp) * (1.0 / c.ny) ** 2 / 4.0
    err = c.T - (Tw + Pr * U * U / (2.0 * cp) * (c.y / h) * (1.0 - c.y / h))
    assert np.ptp(err) < 1e-6 * peak
    assert np.abs(err).max() <= 1.05 * offset
    assert np.ptp(c.p) < 1e-9


# half the work leaves through each wall: conduction alone at the still one, the work in less the conduction out at the moving one
def test_isothermal_couette_passes_the_dissipation_out_through_both_walls(isothermal):
    for name, (F, area) in isothermal.walls.items():
        assert F[4] == pytest.approx(-mu0 * U * U / (2.0 * h) * area, rel=1e-7), name
        assert F[1] == pytest.approx(-mu0 * U / h * area, rel=1e-7), name
