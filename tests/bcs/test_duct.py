import numpy as np
import pytest

import peregrine as pg

from tests.gases import configure, primitives

CELLS, L = 40, 1.0
DX = L / CELLS
P0, T0 = 101325.0, 300.0


def duct(backend, primary, inlet, exit):
    config = pg.Config()
    configure(config, "air", "euler")
    config["adv-flux"]["primary"] = primary
    config["bcvals"]["inlet"] = inlet
    config["bcvals"]["exit"] = exit
    config["bcvals"]["wall"] = {"kind": "slp-adia-wall"}
    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1],
        blkdims=[CELLS + 1, 4, 4],
        lengths=[L, 3 * DX, 3 * DX],
        bcnames={1: "inlet", 2: "exit"},
        defaultbc="wall",
    )
    return pg.get_solver(config, mesh, backend(config))


def uniform(mb, p, u, T):
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    q[..., 0], q[..., 1], q[..., 2:4], q[..., 4] = p, u, 0.0, T
    mb.set_prims([q])


def constants(mb):
    blk = mb.blocks[0]
    fields = list(mb.sim.statefields)
    cell = blk.state.get()[blk.interior][0, 0, 0]
    return tuple(cell[fields.index(f)] for f in ("gamma", "c", "rho"))


FVT = {"kind": "sub-in-fvt", "u": 50.0, "v": 0.0, "w": 0.0, "T": T0}
FP = {"kind": "sub-out-fp", "p": P0}


def pair(backend, name, primary):
    gamma, c, rho = constants(duct(backend, primary, FVT, FP))
    if name == "sub-in-fvt/sub-out-fp":
        return duct(backend, primary, FVT, FP), (P0, FVT["u"], T0)
    if name == "sub-in-ftptt/sub-out-fp":
        M = 0.3
        isen = 1.0 + 0.5 * (gamma - 1.0) * M * M
        inlet = {"kind": "sub-in-ftptt", "pt": P0 * isen ** (gamma / (gamma - 1.0)), "Tt": T0 * isen}
        return duct(backend, primary, inlet, FP), (P0, M * c, T0)
    if name == "sub-in-fmt/sub-out-fp":
        inlet = {"kind": "sub-in-fmt", "mflux": 50.0, "T": T0}
        return duct(backend, primary, inlet, FP), (P0, 50.0 / rho, T0)
    if name == "sup-in-fa/sup-out-fn":
        inlet = {"kind": "sup-in-fa", "p": P0, "u": 650.0, "v": 0.0, "w": 0.0, "T": T0}
        return duct(backend, primary, inlet, {"kind": "sup-out-fn"}), (P0, 650.0, T0)
    raise KeyError(name)


PAIRS = ("sub-in-fvt/sub-out-fp", "sub-in-ftptt/sub-out-fp", "sub-in-fmt/sub-out-fp", "sup-in-fa/sup-out-fn")


def scaled_rhs(mb):
    blk = mb.blocks[0]
    stage = mb.integ.stageregs(mb.regs)[0]
    mb.rhs(Q=stage.Q, dQ=stage.dQ)
    dQ = getattr(blk, stage.dQ).get()[blk.interior]
    Q = getattr(blk, stage.Q).get()[blk.interior]
    _, c, _ = constants(mb)
    u = np.abs(Q[..., 1] / Q[..., 0]).max()
    scale = np.abs(Q).max(axis=(0, 1, 2)) * (u + c) / DX
    # the transverse momentum is zero, so its rows take the axial row's scale
    scale[1:4] = scale[1]
    return np.abs(dQ).max(axis=(0, 1, 2)) / scale


@pytest.mark.parametrize("primary", ["kepaec", "rusanov"])
@pytest.mark.parametrize("name", PAIRS)
def test_the_exact_state_is_a_fixed_point(backend, name, primary):
    mb, (p, u, T) = pair(backend, name, primary)
    uniform(mb, p, u, T)
    assert (scaled_rhs(mb) <= 1e-12).all()
    # a five percent bump of pressure in one cell moves the mass row: the register is live
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    q[blk.ng + CELLS // 2, blk.ng, blk.ng, 0] *= 1.05
    mb.set_prims([q])
    assert scaled_rhs(mb)[0] > 1e-3


def deviations(mb, state, transits):
    blk = mb.blocks[0]
    p, u, T = state
    _, c, _ = constants(mb)
    # half the combined CFL: the duct is three cells across, and at the axial CFL alone rusanov's checkerboard mode outruns rk3
    dt = 0.5 / mb.cflmax()["combined"]
    steps = int(round(L / c / dt))
    out = []
    for _ in range(transits):
        for _ in range(steps):
            mb.step(dt)
        q = primitives(mb, blk)[blk.interior]
        out.append(max(np.abs(q[..., 0] / p - 1.0).max(), np.abs(q[..., 1] / u - 1.0).max(), np.abs(q[..., 4] / T - 1.0).max()))
    return out


# transits to step and the deviation reached by then: the supersonic pair is gone in five (3e-14 measured by ten), the subsonic pairs settle at the flux's own rate (1.8e-5 and 9e-4 measured at thirty)
CONVERGING = [
    ("sup-in-fa/sup-out-fn", 10, 1e-12),
    ("sub-in-ftptt/sub-out-fp", 30, 1e-4),
    ("sub-in-fmt/sub-out-fp", 30, 5e-3),
]


@pytest.mark.parametrize("primary", ["rusanov", "ausmup"])
@pytest.mark.parametrize("name,transits,tol", CONVERGING)
def test_a_dissipative_flux_converges_to_the_exact_state(backend, name, transits, tol, primary):
    mb, (p, u, T) = pair(backend, name, primary)
    uniform(mb, 1.05 * p, 0.95 * u, T)
    assert deviations(mb, (p, u, T), transits)[-1] <= tol


# a velocity inlet and a pressure exit reflect fully at both ends, so only the flux's dissipation drains the standing wave (0.27 to at most 0.11 measured)
@pytest.mark.parametrize("primary", ["rusanov", "ausmup"])
def test_the_resonant_pair_decays(backend, primary):
    mb, (p, u, T) = pair(backend, "sub-in-fvt/sub-out-fp", primary)
    uniform(mb, 1.05 * p, 0.95 * u, T)
    history = deviations(mb, (p, u, T), 30)
    assert max(history[-2:]) < 0.5 * history[0]
