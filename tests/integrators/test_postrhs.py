import numpy as np
import pytest

import peregrine as pg
from peregrine.launch.ranges import CellCenterRange

from tests.gases import hcoeff, primitives
from tests.integrators.cases import box, schemes


# two blocks between walls on i, periodic across j and k: an on-rank interface, periodic halos and wall halos are all made
def case(backend, gas, integrator):
    config, mesh = box(gas=gas, integrator=integrator, dt=1e-5, mbdims=(2, 1, 1), dims=(6, 5, 4), periodic=(False, True, True), names={1: "walls", 2: "walls"})
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    config["ics"]["u"] = 10.0
    mb = pg.get_solver(config, mesh, backend(config))
    prims = []
    for blk in mb.blocks:
        rng = np.random.default_rng(blk.nblki)
        q = primitives(mb, blk)
        q[..., 0] *= 1 + 0.05 * rng.random(q.shape[:3])
        q[..., 1:4] = 30 * rng.random(q.shape[:3] + (3,))
        q[..., 4] *= 1 + 0.1 * rng.random(q.shape[:3])
        prims.append(q)
    mb.set_prims(prims)
    return config, mb


# the interior and the six halo slabs over the interfaces proper; a halo's edges and corners are never written
def slabs(blk):
    for start, extent in CellCenterRange(blk).all():
        yield tuple(slice(s, s + e) for s, e in zip(start, extent))


def check(mb, blk, register):
    Q, S = getattr(blk, register).get(), blk.state.get()
    fields = list(mb.sim.statefields)
    consts = mb.sim.mixture.constants()
    for sl in slabs(blk):
        q, s = Q[sl], S[sl]
        rho = q[..., 0]
        assert np.array_equal(s[..., fields.index("rho")], rho)
        e = (q[..., 4] - 0.5 * (q[..., 1] ** 2 + q[..., 2] ** 2 + q[..., 3] ** 2) / rho) / rho
        Y = q[..., 5:] / rho[..., None]
        Y = np.concatenate([Y, 1.0 - Y.sum(axis=-1)[..., None]], axis=-1)
        R, h298 = Y @ consts["Rsp"], Y @ consts["h298R"]
        T = s[..., fields.index("T")]
        if mb.sim.mixture.eos.name == "cpg":
            cp = Y @ consts["cp0"]
            # measured 2.9e-16 and 1.9e-16
            assert np.allclose(T, (e - h298) / (cp - R), rtol=1e-14, atol=0.0)
            assert np.allclose(s[..., fields.index("p")], rho * R * T, rtol=1e-14, atol=0.0)
        else:
            # h = T sum_m hcoeff_m (ln T)^m + h298 on the mixture's cp moment, the coefficients by the kernel's recurrence and its Horner
            coeff = hcoeff(Y @ consts["cppolyR"])
            u = np.log(T)
            h = np.zeros_like(T)
            for m in reversed(range(coeff.shape[-1])):
                h = h * u + coeff[..., m]
            h = h * T + h298
            cp = s[..., fields.index("cp")]
            # the fits cancel large terms, so the residual the Newton leaves is seen through their rounding: measured 5.2e-12 of cp T; a temperature one Newton step short reads 1e-6
            assert (np.abs((h - R * T) - e) / (cp * T) < 1e-10).all()
            assert np.allclose(s[..., fields.index("p")], rho * R * T, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("gas", ["air", "CH4_O2"])
@pytest.mark.parametrize("integrator", schemes)
def test_every_stage_register_is_fit_to_read(backend, gas, integrator):
    config, mb = case(backend, gas, integrator)
    mb.postrhs()
    for blk in mb.blocks:
        check(mb, blk, "Q")
    # the stages by hand, each register readied after its advance, since a step leaves only the last stage's register readable
    mb.dt.set(config["simulation"]["dt"])
    for n, stage in enumerate(mb.integ.stageregs(mb.regs)):
        mb.rhs(Q=stage.Q, dQ=stage.dQ)
        mb.advance(n)
        mb.prerhs(stage.Qout, stage.dQ)
        for blk in mb.blocks:
            check(mb, blk, stage.Qout)
            # the banks are told apart: a stage register well off the solution, or the solution itself moved
            if stage.Qout != "Q":
                apart = np.abs(blk.Q1.get() - blk.Q.get())[blk.interior].max() / np.abs(blk.Q.get()[blk.interior]).max()
                assert apart > 1e-8
