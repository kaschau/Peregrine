import numpy as np
import pytest

import peregrine as pg
from peregrine.config import ConfigError

from tests.gases import primitives

P0, CELLS, DX = 59.8e6, 40, 5e-5
SPECIES = ["CH4", "O2"]


def config(doubleflux, simulator="euler", **advflux):
    config = pg.Config()
    config["mixture"].update(species=SPECIES, eos="real-gas-df")
    config["mixture"]["T-range"] = (100.0, 1500.0)
    config["simulation"]["simulator"] = simulator + ("-df" if doubleflux else "")
    config["adv-flux"].update(advflux)
    config["time-integration"]["integrator"] = "tvd-rk3"
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    return config


def box(cells=CELLS, dx=DX):
    return pg.mesher.CubeMesher(
        mbdims=[1, 1, 1],
        blkdims=[cells + 1, 2, 2],
        lengths=[cells * dx, dx, dx],
        periodic=[True, False, False],
        bcnames={3: "walls", 4: "walls", 5: "walls", 6: "walls"},
    )


def solver(backend, doubleflux, **advflux):
    cfg = config(doubleflux, **advflux)
    return pg.get_solver(cfg, box(), backend(cfg))


def interior(mb):
    ng = mb.blocks[0].ng
    return (slice(ng, -ng), ng, ng)


def field(mb, name, where):
    return mb.blocks[0].state.get()[where][..., mb.sim.statefields.index(name)]


# tanh edges, so the central flux carries a profile rather than a jump
def slab(mb, u0=20.0, Tin=162.7, Tout=111.7, smear=1.5, cells=CELLS):
    blk = mb.blocks[0]
    ng = blk.ng
    q = primitives(mb, blk)
    i = np.arange(q.shape[0]) - ng + 0.5
    f = 0.5 * (np.tanh((i - cells // 4) / smear) - np.tanh((i - cells // 2) / smear))
    f = f[:, None, None]
    q[..., 0], q[..., 1:4] = P0, 0.0
    q[..., 1] = u0
    q[..., 4] = Tout + f * (Tin - Tout)
    q[..., 5] = f
    mb.set_prims([q])
    return interior(mb)


def relative(change, totals):
    return np.divide(change, np.abs(totals), out=np.zeros_like(change), where=totals != 0)


def advect(mb, cfl, steps, u0=20.0):
    blk = mb.blocks[0]
    inner = slab(mb, u0=u0)
    dt = cfl * DX / field(mb, "c", inner).max()
    Q0 = blk.Q.get()[inner]
    totals0 = Q0.sum(axis=0)
    scatter = 0.0
    for _ in range(steps):
        mb.step(dt)
        p = field(mb, "p", inner)
        assert np.all(np.isfinite(p))
        scatter = max(scatter, np.abs(p / P0 - 1.0).max())
    Q = blk.Q.get()[inner]
    drift = relative(Q.sum(axis=0) - totals0, totals0)
    moved = int(round(u0 * dt * steps / DX))
    shape = np.abs(Q[:, 5] / Q[:, 0] - np.roll(Q0[:, 5] / Q0[:, 0], moved)).max()
    return scatter, drift, shape


def test_a_real_gas_contact_keeps_its_pressure_at_the_schemes_cfl(backend):
    # measured: the plain flux scatters the pressure by 4.7e-3 at CFL 0.4
    plain, _, _ = advect(solver(backend, False), 0.4, 200)
    scatter, drift, shape = advect(solver(backend, True), 1.2, 400)
    assert plain > 1e-3 and scatter < 1e-9
    assert abs(drift[0]) < 1e-12 and abs(drift[5]) < 1e-12
    # the shape error is the central scheme's dispersion on the edges, measured 0.19 of the fraction
    assert shape < 0.25
    # the frozen planes' energy drift is second order in a step's change of state: half the step drops it by two to four (measured 0.36)
    assert abs(drift[4]) < 2e-3
    _, halved, _ = advect(solver(backend, True), 0.6, 400)
    assert abs(halved[4]) < 0.5 * abs(drift[4])


@pytest.mark.parametrize(
    "primary, recon, limiter, plainabove",
    [
        ("rusanov", "constant", None, 1e-3),
        ("ausmup", "constant", None, 1e-4),
        ("rusanov", "muscl", "koren", 1e-5),
        ("ausmup", "muscl", "koren", 1e-5),
    ],
    ids=["rusanov", "ausmup", "muscl-koren-rusanov", "muscl-koren-ausmup"],
)
def test_the_riemann_solvers_keep_the_pressure_too(backend, primary, recon, limiter, plainabove):
    # with MUSCL both sides' energies are formed from the reconstructed states in each cell's plane, the eos's energy of a reconstructed state being off the plane
    flux = dict(primary=primary, recon=recon, limiter=limiter)
    plain, _, _ = advect(solver(backend, False, **flux), 0.4, 200)
    scatter, drift, _ = advect(solver(backend, True, **flux), 0.4, 200)
    assert plain > plainabove and scatter < 1e-9
    assert abs(drift[0]) < 1e-12 and abs(drift[5]) < 1e-12


def test_the_switched_blend_keeps_the_pressure(backend):
    # both fluxes write the energy per side, so their blend does; gamma-gain 0 leaves the pressure sensor switching alone
    switched = dict(formulation="blended", secondary="rusanov", switch="jameson", values={"p-gain": 5.0, "gamma-gain": 0.0})
    scatter, drift, _ = advect(solver(backend, True, **switched), 0.4, 200)
    assert scatter < 1e-9
    assert abs(drift[0]) < 1e-12 and abs(drift[5]) < 1e-12


def test_the_energy_row_is_added_once_over_a_begun_derivative(backend):
    # cold enough that the mechanism's rates are nothing, so the right-hand side with chemistry is the one without
    mechanism = "CH4_O2_FFCMY.yaml"

    def rhs(kind):
        config = pg.Config()
        config["mixture"].update(species=mechanism, eos="real-gas-df")
        config["mixture"]["T-range"] = (100.0, 1500.0)
        config["simulation"]["simulator"] = "euler-df"
        config["mixture"]["chemistry"] = "finite-rate" if kind else None
        config["chem-integration"]["kind"] = kind
        config["time-integration"]["integrator"] = "tvd-rk3"
        config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
        mb = pg.get_solver(config, box(), backend(config))
        blk = mb.blocks[0]
        names = mb.sim.mixture.spnames
        ng = blk.ng
        q = primitives(mb, blk)
        i = np.arange(q.shape[0]) - ng + 0.5
        f = 0.5 * (np.tanh((i - CELLS // 4) / 1.5) - np.tanh((i - CELLS // 2) / 1.5))[:, None, None]
        q[...] = 0.0
        q[..., 0], q[..., 1] = P0, 20.0
        q[..., 4] = 111.7 + f * (162.7 - 111.7)
        q[..., 5 + names.index("CH4")] = f
        q[..., 5 + names.index("O2")] = 1.0 - f
        mb.set_prims([q])
        mb.dt.set(1e-8)
        stage = mb.integ.stageregs(mb.regs)[0]
        mb.rhs(Q=stage.Q, dQ=stage.dQ)
        return getattr(blk, stage.dQ).get()[interior(mb)]

    plain, begun = rhs(None), rhs("coupled")
    scale = np.abs(plain).max(axis=0)
    assert np.all(np.abs(begun - plain).max(axis=0) <= 1e-12 * np.where(scale > 0, scale, 1.0))


@pytest.mark.parametrize("eos", ["tpg", "real-gas"])
def test_it_refuses_an_eos_without_a_plane(eos):
    config = pg.Config()
    config["mixture"].update(species=SPECIES, eos=eos)
    config["mixture"]["T-range"] = (300.0, 1500.0)
    config["simulation"]["simulator"] = "euler-df"
    with pytest.raises(ConfigError, match="tangent plane"):
        pg.simulator.get_simulator(config)


def pulsespeed(backend, doubleflux, cells=400):
    cfg = config(doubleflux)
    mb = pg.get_solver(cfg, box(cells), backend(cfg))
    blk = mb.blocks[0]
    ng = blk.ng
    q = primitives(mb, blk)
    i = np.arange(q.shape[0]) - ng + 0.5
    q[..., 0] = (P0 * (1.0 + 1e-3 * np.exp(-(((i - cells / 2) / 6.0) ** 2))))[:, None, None]
    q[..., 1:4], q[..., 4], q[..., 5] = 0.0, 111.7, 0.0
    mb.set_prims([q])
    inner = interior(mb)
    c = float(np.median(field(mb, "c", inner)))
    dt = 0.4 * DX / c
    x = (np.arange(cells) + 0.5) * DX
    right = slice(cells // 2 + 10, cells)

    def peak():
        dp = field(mb, "p", inner) - P0
        return x[right][np.argmax(dp[right])]

    for _ in range(60):
        mb.step(dt)
    x1 = peak()
    for _ in range(100):
        mb.step(dt)
    return (peak() - x1) / (100 * dt), c


def test_a_pulse_keeps_the_speed_of_sound(backend):
    # a plane tangent in the pressure alone would put a liquid's sound speed at a third of the eos's
    plain, c = pulsespeed(backend, False)
    double, _ = pulsespeed(backend, True)
    assert abs(plain / c - 1.0) < 0.05
    assert abs(double / plain - 1.0) < 0.01


def ignition(backend, doubleflux, dt=1e-8, steps=400):
    config = pg.Config()
    config["mixture"].update(species="CH4_O2_FFCMY.yaml", eos="real-gas-df")
    config["mixture"]["T-range"] = (300.0, 4000.0)
    config["simulation"].update(simulator="euler-df" if doubleflux else "euler", controller="fixed", dt=dt)
    config["mixture"]["chemistry"] = "finite-rate"
    config["chem-integration"].update(kind="sub-stepped")
    config["chem-integration"]["max-substeps"] = 10
    config["time-integration"]["integrator"] = "tvd-rk3"
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    mesh = pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[3, 3, 3], lengths=[1e-3] * 3, defaultbc="walls")
    mb = pg.get_solver(config, mesh, backend(config))
    blk = mb.blocks[0]
    names = mb.sim.mixture.spnames
    q = primitives(mb, blk)
    q[...] = 0.0
    q[..., 0], q[..., 4] = 5e6, 2000.0
    q[..., 5 + names.index("CH4")] = 0.2
    q[..., 5 + names.index("O2")] = 0.8
    mb.set_prims([q])
    for _ in range(steps):
        mb.step(dt)
    ng = blk.ng
    where = (ng, ng, ng)
    return field(mb, "T", where), field(mb, "p", where)


def test_a_closed_box_ignites_to_the_plain_schemes_state(backend):
    # what the freeze drops is second order in the composition change a step makes, and shrinks with the step
    Tplain, pplain = ignition(backend, False)
    T, p = ignition(backend, True)
    assert Tplain > 4000.0
    assert abs(T / Tplain - 1.0) < 5e-3 and abs(p / pplain - 1.0) < 1e-2


def diffusing(backend, doubleflux, dx=1e-7, steps=2000):
    config = pg.Config()
    config["mixture"].update(species=SPECIES, eos="real-gas-df", trans="kinetic-theory")
    config["mixture"]["T-range"] = (100.0, 1500.0)
    config["simulation"]["simulator"] = "navier-stokes-df" if doubleflux else "navier-stokes"
    config["time-integration"]["integrator"] = "tvd-rk3"
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    mb = pg.get_solver(config, box(dx=dx), backend(config))
    blk = mb.blocks[0]
    inner = slab(mb, u0=0.0, Tin=140.0, Tout=140.0, smear=3.0)
    dt = 0.4 * dx / field(mb, "c", inner).max()
    E0 = blk.Q.get()[inner][:, 4].sum()
    for _ in range(steps):
        mb.step(dt)
    drift = (blk.Q.get()[inner][:, 4].sum() - E0) / abs(E0)
    return field(mb, "T", inner), field(mb, "p", inner), drift


def test_a_diffusing_layer_keeps_its_temperature(backend):
    # 2000 steps let the edges mix a few thousandths, where a plane without composition slopes reads a kelvin of heating
    # a real fluid's volume of mixing moves the pressure by 1e-4 P0 under the plain physics too, so it is held to the plain's, not to uniform
    Tplain, pplain, _ = diffusing(backend, False)
    T, p, drift = diffusing(backend, True)
    assert np.abs(T - Tplain).max() < 0.1
    assert np.abs(p - pplain).max() / P0 < 1e-4
    assert abs(drift) < 1e-4


def heated(backend, doubleflux, Tbox=300.0, Twall=400.0, steps=0):
    config = pg.Config()
    config["mixture"].update(species=SPECIES, eos="real-gas-df", trans="kinetic-theory")
    config["mixture"]["T-range"] = (100.0, 1500.0)
    config["simulation"]["simulator"] = "navier-stokes-df" if doubleflux else "navier-stokes"
    config["time-integration"]["integrator"] = "tvd-rk3"
    config["bcvals"]["walls"] = {"kind": "no-slp-isot-wall", "T": Twall}
    mesh = pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[6, 5, 4], lengths=[2.5e-4, 2e-4, 1.5e-4], defaultbc="walls")
    mb = pg.get_solver(config, mesh, backend(config))
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    q[..., 0], q[..., 1:4], q[..., 4], q[..., 5] = P0, 0.0, Tbox, 0.5
    mb.set_prims([q])
    dt = 0.3 * 5e-5 / field(mb, "c", blk.interior).max()
    for _ in range(steps):
        mb.step(dt)
    stage = mb.integ.stageregs(mb.regs)[0]
    mb.rhs(Q=stage.Q, dQ=stage.dQ)
    dQ = getattr(blk, stage.dQ).get()[blk.interior][..., 4]
    return [(np.take(dQ, 0, axis=a), np.take(dQ, -1, axis=a)) for a in range(3)]


def test_a_wall_heats_both_sides_of_the_box_alike(backend):
    plain, df = heated(backend, False), heated(backend, True)
    for (low, high), (lowdf, highdf) in zip(plain, df):
        assert np.allclose(lowdf, low, rtol=1e-12)
        assert np.allclose(highdf, high, rtol=1e-12)
        assert np.allclose(lowdf.mean(), highdf.mean(), rtol=1e-12)


def test_a_hot_box_against_cold_walls_stays_finite(backend):
    # a mirrored halo at 2 Twall - Tbox would have no real-gas state; the halo holds the wall's own
    for low, high in heated(backend, True, Tbox=1400.0, Twall=300.0, steps=30):
        assert np.isfinite(low).all() and np.isfinite(high).all()
        assert (low < 0.0).all() and (high < 0.0).all()
