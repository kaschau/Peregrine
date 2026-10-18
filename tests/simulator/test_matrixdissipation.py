import numpy as np

from tests.gases import primitives
from tests.minimal.cases import JostledCube
from tests.simulator import oracle as orc
from tests.simulator import support as sp
from tests.simulator.test_facefluxes import close, fluxed, sides, slab, smooth

VALUES = {"p-gain": 5.0, "gamma-gain": 1.0}
MATRIX = dict(formulation="correction", primary="kepaec", recon="constant", switch="jameson", correction="matrix-dissipation")


def corrected(backend, values=VALUES, **kw):
    return sp.box(backend, **MATRIX, values=values, **kw)


# this jostle keeps the block's outside planes, so the grid is the same at either halo depth and its periodic faces telescope
def periodicbox(backend, **adv):
    return sp.case(backend, (7, 6, 5), [1.0, 1.0, 1.0], [True] * 3, gas=sp.binary, mesher=JostledCube, **adv)


# drawn over the interior alone, so the state is the same at either halo depth
def randomstate(mb, seed=11):
    blk = mb.blocks[0]
    rng = np.random.default_rng(seed)
    q = primitives(mb, blk)
    inner = q[sp.interior(blk)]
    shape = inner.shape[:3]
    inner[..., 0] = 101325.0 * (1.0 + 0.2 * rng.uniform(-1, 1, shape))
    inner[..., 1] = 60.0 * rng.uniform(-1, 1, shape)
    inner[..., 2] = 40.0 * rng.uniform(-1, 1, shape)
    inner[..., 3] = 30.0 * rng.uniform(-1, 1, shape)
    inner[..., 4] = 300.0 * (1.0 + 0.3 * rng.uniform(-1, 1, shape))
    inner[..., 5] = 0.5 + 0.3 * rng.uniform(-1, 1, shape)
    mb.set_prims([q])


def sensors(f, values=VALUES):
    return orc.jameson(f.state, pgain=values["p-gain"], ggain=values["gamma-gain"])


def correction(f, mb, shock, contact):
    L, R = sides(f, "constant")
    added = f.F[..., : mb.sim.ne] - orc.kepaec(L, R, f.S)
    return added, orc.matrixdissipation(sp.binary, L, R, f.S, shock, contact)


def test_the_face_flux_is_kepaec_plus_the_correction(backend):
    mb = corrected(backend)
    randomstate(mb)
    fluxed(mb)
    for axis in range(3):
        f = sp.faces(mb, axis, 2)
        shock, contact = sensors(f)
        # the sensors spread from low to well on, so the weighing is exercised at every level
        assert shock.min() < 0.2 and shock.max() > 0.5
        got, want = correction(f, mb, shock, contact)
        # measured at most 3e-14 of a row's largest, kepaec's own rounding included
        close(got, want, 1e-12)


def test_the_pressure_and_velocity_rows_take_the_shock_sensor_alone(backend):
    # a composition jump at the smooth state's pressure: the contact sensor clips to one at the jump where the shock sensor stays small, and the rows weighed by the larger sensor alone would not match
    values = {"p-gain": 5.0, "gamma-gain": 1000.0}
    mb = corrected(backend, values)
    slab(mb, 5, 0.1, 0.9)
    fluxed(mb)
    f = sp.faces(mb, 0, 2)
    shock, contact = sensors(f, values)
    assert contact.max() == 1.0 and shock.max() < 0.5
    got, want = correction(f, mb, shock, contact)
    close(got, want, 1e-12)
    larger = orc.matrixdissipation(sp.binary, *sides(f, "constant"), f.S, contact, contact)
    scale = np.abs(want.reshape(-1, mb.sim.ne)).max(axis=0)
    assert (np.abs(larger - want).reshape(-1, mb.sim.ne).max(axis=0)[:5] > 1e-3 * scale[:5]).all()


def test_sensors_off_leave_kepaec_bitwise(backend):
    # the same grid and interior state under either halo depth
    plain = periodicbox(backend, **sp.schemes["kepaec"])
    off = periodicbox(backend, **MATRIX, values={"p-gain": 0.0, "gamma-gain": 0.0})
    for mb in (plain, off):
        randomstate(mb)
        fluxed(mb)
    for axis in range(3):
        assert np.array_equal(sp.faces(off, axis).F[..., : off.sim.ne], sp.faces(plain, axis).F)


def test_the_flux_is_skew_symmetric_under_a_mirror(backend):
    # the line reversed cell for cell with u negated: the sensors are even under it and the correction's rows negate with the mass, cross momentum, energy and species rows while the normal momentum keeps
    mb = sp.line(backend, 41, gas=sp.binary, **MATRIX, values=VALUES)
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
    assert (np.abs(Fm[::-1] - signs * F) <= 1e-13 * scale).all()


def test_the_periodic_box_conserves(backend):
    # the correction is a face flux, so the derivative over the box telescopes to nothing in every row; measured 2e-15 of the row's largest face flux
    mb = periodicbox(backend, **MATRIX, values=VALUES)
    randomstate(mb)
    stage = sp.stepregs(mb)
    mb.rhs(Q=stage.Q, dQ=stage.dQ)
    blk = mb.blocks[0]
    inner = sp.interior(blk)
    dQ = getattr(blk, stage.dQ).get()[inner]
    total = (dQ / blk.detJinv.get()[inner][..., None]).sum(axis=(0, 1, 2))
    scale = np.max([np.abs(getattr(blk, f"{a}F").get()[inner]).max(axis=(0, 1, 2)) for a in "ijk"], axis=0)
    assert (np.abs(total) <= 1e-12 * scale).all()


def test_the_wave_keeps_kepaecs_numbers(backend):
    # the wave's pressure and velocity are uniform to rounding and its gamma one gas's, so neither sensor fires beyond rounding and the flux is kepaec's; measured within 1e-12
    alone = sp.wave_errors(backend, 41, **sp.schemes["kepaec"])
    assert np.allclose(sp.wave_errors(backend, 41, **MATRIX, values=VALUES), alone, rtol=1e-11, atol=0.0)


def test_sod_is_sharper_than_kepaec_and_about_the_blend(backend):
    # the central scheme alone rings across Sod's tube; the correction damps the ringing where the sensor fires, to a third of kepaec's density error (measured 0.39) and within a fifth of the blend's (measured 1.16)
    alone = sp.sod(backend, **sp.schemes["kepaec"]).rho_err
    blend = sp.sod(backend, **sp.schemes["kepaec+rusanov by jameson"]).rho_err
    matrix = sp.sod(backend, **MATRIX, values=VALUES).rho_err
    assert matrix < 0.6 * alone
    assert matrix < 1.3 * blend


def test_the_halo_reaches_the_switchs_cells(backend):
    # two cells either side for Jameson's second difference, whatever the reconstruction
    assert corrected(backend).ng == 2


def test_it_rides_on_the_smooth_state_too(backend):
    # the sensors small everywhere: the correction a fraction of the flux (measured under a third of every row), and still the transcription
    mb = corrected(backend)
    smooth(mb)
    fluxed(mb)
    for axis in range(3):
        f = sp.faces(mb, axis, 2)
        shock, contact = sensors(f)
        assert shock.max() < 0.5
        got, want = correction(f, mb, shock, contact)
        close(got, want, 1e-12)
