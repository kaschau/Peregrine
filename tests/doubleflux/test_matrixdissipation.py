import numpy as np

from tests.doubleflux.test_doubleflux import DX, P0, advect, field, slab, solver

MATRIX = dict(formulation="correction", correction="matrix-dissipation", switch="jameson", values={"p-gain": 5.0, "gamma-gain": 10.0})


def test_the_correction_keeps_the_pressure_too(backend):
    # the energy row per side in that cell's plane, the rest one flux the relation's derivatives make: measured 1.0e-11 against the plain physics' 4.0e-3
    plain, _, _ = advect(solver(backend, False, **MATRIX), 0.4, 200)
    scatter, drift, _ = advect(solver(backend, True, **MATRIX), 0.4, 200)
    assert plain > 1e-3 and scatter < 1e-9
    assert abs(drift[0]) < 1e-12 and abs(drift[5]) < 1e-12


def overshoot(mb, cfl=0.8, steps=400):
    blk = mb.blocks[0]
    inner = slab(mb, smear=1.5)
    dt = cfl * DX / field(mb, "c", inner).max()
    scatter = 0.0
    for _ in range(steps):
        mb.step(dt)
        p = field(mb, "p", inner)
        assert np.all(np.isfinite(p))
        scatter = max(scatter, np.abs(p / P0 - 1.0).max())
    Q = blk.Q.get()[inner]
    Y = Q[:, 5] / Q[:, 0]
    return max(Y.max() - 1.0, -Y.min()), scatter


def test_the_correction_damps_the_contacts_overshoot(backend):
    # switched on by gamma changing across the contact, the correction damps the species and the temperature at the flow speed and leaves the pressure kept: the overshoot halves, 0.112 to 0.049 measured
    alone, _ = overshoot(solver(backend, True))
    damped, scatter = overshoot(solver(backend, True, **MATRIX))
    assert scatter < 1e-9
    assert damped < 0.5 * alone and damped < 0.06
