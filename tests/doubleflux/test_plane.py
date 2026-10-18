import numpy as np
import pytest

import peregrine as pg

from tests.gases import primitives

P0, T0, CELLS = 59.8e6, 130.0, 40
SPECIES = ["CH4", "O2"]


def solver(backend, simulator):
    config = pg.Config()
    config["mixture"].update(species=SPECIES, eos="real-gas-df")
    config["mixture"]["T-range"] = (100.0, 1500.0)
    config["simulation"]["simulator"] = simulator
    config["time-integration"]["integrator"] = "tvd-rk3"
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1],
        blkdims=[CELLS + 1, 2, 2],
        lengths=[CELLS * 5e-5, 5e-5, 5e-5],
        periodic=[True, False, False],
        bcnames={3: "walls", 4: "walls", 5: "walls", 6: "walls"},
    )
    return pg.get_solver(config, mesh, backend(config))


# set_prims settles the state by the eos and freezes the plane
def setstate(mb, p, T, Y0, u=20.0):
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    q[..., 0], q[..., 1:4], q[..., 4], q[..., 5] = p, 0.0, T, Y0
    q[..., 1] = u
    mb.set_prims([q])


def fields(mb, where):
    state = mb.blocks[0].state.get()[where]
    return {name: state[..., k] for k, name in enumerate(mb.sim.statefields)}


def plane_coords(mb, where):
    Q = mb.blocks[0].Q.get()[where]
    return Q[..., 0] * fields(mb, where)["e"], Q[..., 0], Q[..., 5]


@pytest.fixture(scope="module")
def df(backend):
    return solver(backend, "euler-df")


def test_the_plane_is_tangent_to_the_eos(df):
    # the residual's difference across the perturbation is second order in h, the energy's own first
    ng = df.blocks[0].ng
    cell = (ng, ng, ng)
    setstate(df, P0, T0, 0.5)
    plane = {name: fields(df, cell)[name] for name in ("theta", "eps", "kappa", "zeta0")}

    def residual(p, T, Y0):
        setstate(df, p, T, Y0)
        E, rho, rhoY0 = plane_coords(df, cell)
        return E - (plane["theta"] * p + plane["eps"] * rho + plane["zeta0"] * rhoY0 + plane["kappa"]), E

    h = 1e-3
    # the three primitive directions span the plane's (p, rho, rho Y0) space, so tangency along them is tangency everywhere
    for direction in ("p", "T", "Y0"):
        up = dict(p=P0, T=T0, Y0=0.5)
        down = dict(up)
        up[direction] *= 1.0 + h
        down[direction] *= 1.0 - h
        Rup, Eup = residual(**up)
        Rdown, Edown = residual(**down)
        assert abs(Rup - Rdown) < 1e-5 * abs(Eup - Edown), direction


def test_the_frozen_state_is_the_eoss(backend, df):
    # the set state has already been through the plane's pressure and the eos at it
    plain = solver(backend, "euler")
    setstate(df, P0, T0, 0.5)
    setstate(plain, P0, T0, 0.5)
    ng = df.blocks[0].ng
    inner = (slice(ng, -ng), ng, ng)
    eos = plain.sim.statefields
    ours, theirs = fields(df, inner), fields(plain, inner)
    for name in eos:
        assert np.allclose(ours[name], theirs[name], rtol=1e-12, atol=0.0), name
    df.postrhs()
    again = fields(df, inner)
    for name in df.sim.statefields:
        assert np.allclose(again[name], ours[name], rtol=1e-12, atol=0.0), name


def test_the_halos_carry_their_images_planes(df):
    # to rounding rather than bitwise: a halo's real-gas Newton starts from its own prims' guess
    blk = df.blocks[0]
    ng = blk.ng
    setstate(df, P0, T0, 0.5)
    state = blk.state.get()[:, ng, ng, :]
    for layer in range(ng):
        low, lowimage = state[layer], state[CELLS + layer]
        high, highimage = state[CELLS + ng + layer], state[ng + layer]
        assert np.allclose(low, lowimage, rtol=1e-12, atol=0.0)
        assert np.allclose(high, highimage, rtol=1e-12, atol=0.0)
