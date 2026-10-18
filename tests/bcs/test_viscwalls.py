import numpy as np

import peregrine as pg

from tests.gases import primitives

N, H = 16, 1e-3
DY = H / N
CP, MW = 1005.0, 28.96
# a diffusivity half of a cell's acoustic reach per unit time, at Prandtl one
MU0 = 0.5 * DY * 347.0 * 1.17
KAPPA0 = MU0 * CP
GAS = {"Air": {"MW": MW, "cp0": CP, "mu0": MU0, "kappa0": KAPPA0}}
T0, U = 300.0, 100.0


def column(backend, low, high, u0=0.0):
    config = pg.Config()
    config["simulation"]["simulator"] = "navier-stokes"
    config["mixture"]["species"] = GAS
    config["mixture"]["eos"] = "cpg"
    config["mixture"]["trans"] = "constant"
    config["bcvals"]["low"], config["bcvals"]["high"] = low, high
    config["ics"]["u"] = u0
    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1], blkdims=[2, N + 1, 2], lengths=[DY, H, DY], periodic=[True, False, True], bcnames={3: "low", 4: "high"}
    )
    return pg.get_solver(config, mesh, backend(config))


# the column's fundamental acoustic wave, which only viscosity damps, decays twenty-six fold per thousand steps, so eight thousand reach roundoff
def settle(mb, steps=8000):
    blk = mb.blocks[0]
    c = list(mb.sim.statefields).index("c")
    # three tenths of the acoustic CFL also keeps the diffusion stable
    dt = 0.3 * DY / blk.state.get()[blk.interior][..., c].max()
    for _ in range(steps):
        mb.step(dt)
    q = primitives(mb, blk)[blk.interior][0, :, 0, :]
    y = blk.xcc.get()[blk.interior][0, :, 0, 1]
    return q, y


def wallflux(mb, row):
    blk = mb.blocks[0]
    low, high = blk.get_intface(3), blk.get_intface(4)
    F = blk.jF.get()
    S = np.linalg.norm(low.faceplane(blk.jS.get()), axis=-1)
    return low.faceplane(F)[..., row], high.faceplane(F)[..., row], S


def couette(backend):
    return column(backend, {"kind": "no-slp-isot-wall", "T": T0}, {"kind": "mov-isot-wall", "u": U, "v": 0.0, "w": 0.0, "T": T0}, u0=0.5 * U)


# linear profiles are exact to roundoff: the alpha-damping gradients are exact on them and the wall halos are their linear extrapolation
def test_couette_flow_is_linear(backend):
    mb = couette(backend)
    q, y = settle(mb)
    assert np.allclose(q[:, 1], U * y / H, rtol=0.0, atol=1e-12 * U)
    assert np.allclose(q[:, 2:4], 0.0, atol=1e-12 * U)
    # the shear drags the fluid back at the still wall and along at the moving one
    low, high, S = wallflux(mb, 1)
    assert np.allclose(low, -MU0 * U / H * S, rtol=1e-12)
    assert np.allclose(high, -MU0 * U / H * S, rtol=1e-12)


def test_couette_flow_heats_the_fluid(backend):
    mb = couette(backend)
    q, y = settle(mb)
    peak = MU0 * U * U / (8.0 * KAPPA0)
    want = T0 + MU0 * U * U / (2.0 * KAPPA0) * (y / H) * (1.0 - y / H)
    # within the two-point wall flux's shift of a 1/N^2 of the peak; the dissipation mu U^2 / h leaves half through each wall
    assert np.abs(q[:, 4] - want).max() <= 2.0 / N**2 * peak
    low, high, S = wallflux(mb, 4)
    assert np.allclose(low, -MU0 * U * U / (2.0 * H) * S, rtol=1e-10)
    assert np.allclose(high, -MU0 * U * U / (2.0 * H) * S, rtol=1e-10)


def test_conduction_is_linear(backend):
    T1, T2 = 300.0, 400.0
    mb = column(backend, {"kind": "no-slp-isot-wall", "T": T1}, {"kind": "no-slp-isot-wall", "T": T2})
    q, _ = settle(mb)
    want = T1 + (T2 - T1) * (np.arange(N) + 0.5) / N
    assert np.allclose(q[:, 4], want, rtol=0.0, atol=1e-12 * (T2 - T1))
    assert np.allclose(q[:, 1:4], 0.0, atol=1e-11)
    low, high, S = wallflux(mb, 4)
    assert np.allclose(low, -KAPPA0 * (T2 - T1) / H * S, rtol=1e-12)
    assert np.allclose(high, -KAPPA0 * (T2 - T1) / H * S, rtol=1e-12)
