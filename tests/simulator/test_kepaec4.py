import numpy as np
import pytest

from tests.gases import primitives
from tests.minimal.cases import P0, U0, carry_wave
from tests.simulator import oracle as orc
from tests.simulator import support as sp
from tests.simulator.test_facefluxes import close, fluxed, smooth

flux = dict(primary="kepaec4", recon="constant")


def test_the_face_flux_is_its_formula(backend):
    mb = sp.box(backend, **flux)
    smooth(mb)
    fluxed(mb)
    for axis in range(3):
        f = sp.faces(mb, axis, 2)
        cells = [orc.cell(f.Q[o], f.state[o]) for o in (-2, -1, 0, 1)]
        # measured at most 6.6e-16 of a row's largest
        close(f.F[..., : mb.sim.ne], orc.kepaec4(*cells, f.S), 1e-13)


def test_the_halo_is_two_deep(backend):
    # the outer pairs read the cell beyond each side
    assert sp.line(backend, 11, **flux).ng == 2


def test_the_flux_is_skew_symmetric_under_a_mirror(backend):
    # the line reversed cell for cell with u negated: face f becomes face N - f, and its flux's mass, cross momentum, energy and species rows negate while the normal momentum keeps
    mb = sp.line(backend, 41, gas=sp.binary, **flux)
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


def test_the_wave_errs_alike_leftward_and_rightward(backend):
    # a central scheme has no upwind side to bias: measured within 5e-11 of the error, which is 3e-15 of the density
    forward, backward = sp.wave_errors(backend, 41, u=100.0, **flux), sp.wave_errors(backend, 41, u=-100.0, **flux)
    assert np.allclose(backward, forward, rtol=1e-9, atol=0.0)


def test_the_wave_keeps_the_pressure_and_the_velocity(backend):
    # each pair's thermal energy flux is p U / (gamma - 1) whatever its densities and the three weights sum to one, so the combination keeps the two-point flux's pressure equilibrium
    error, prims, drift = carry_wave(backend, 41, **flux)
    assert np.abs(prims[:, 0] / P0 - 1.0).max() < 1e-12
    assert np.abs(prims[:, 1] / U0 - 1.0).max() < 1e-12
    assert abs(drift) < 1e-12


def test_it_takes_the_constant_reconstruction_only(backend):
    # a reconstructed side is no cell, so the outer pairs would be of nothing: refused as the kernel is rendered, before any arithmetic
    with pytest.raises(ValueError, match="kepaec4 takes the constant reconstruction"):
        sp.line(backend, 11, primary="kepaec4", recon="muscl", limiter="koren")
