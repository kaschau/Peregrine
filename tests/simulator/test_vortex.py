import numpy as np
import pytest

from tests.gases import primitives
from tests.simulator import support as sp

# R = 1 and gamma = 1.4: Ru / MW = 1, cp0 = gamma / (gamma - 1)
gas = {"G": {"MW": sp.Ru, "cp0": 3.5}}
gamma, beta, side = 1.4, 5.0, 10.0


def exact(x, y, t):
    dx = (x - 5.0 - t + side / 2) % side - side / 2
    dy = y - 5.0
    r2 = dx**2 + dy**2
    f = np.exp(0.5 * (1.0 - r2))
    T = 1.0 - (gamma - 1.0) * beta**2 / (8.0 * gamma * np.pi**2) * np.exp(1.0 - r2)
    rho = T ** (1.0 / (gamma - 1.0))
    return dict(p=rho * T, u=1.0 - beta / (2.0 * np.pi) * dy * f, v=beta / (2.0 * np.pi) * dx * f, T=T, rho=rho)


def vortex(backend, n, plane=(0, 1), **adv):
    a, b = plane
    dims, lengths = [2, 2, 2], [side / (n - 1)] * 3
    dims[a] = dims[b] = n
    lengths[a] = lengths[b] = side
    mb = sp.case(backend, dims, lengths, [True] * 3, gas=gas, **adv)
    blk = mb.blocks[0]
    X = blk.xcc.get()
    start = exact(X[..., a], X[..., b], 0.0)
    q = primitives(mb, blk)
    q[..., 0], q[..., 4] = start["p"], start["T"]
    q[..., 1 + a], q[..., 1 + b] = start["u"], start["v"]
    mb.set_prims([q])
    dx = side / (n - 1)
    steps = int(round(side / (0.3 * dx / (1.0 + np.sqrt(gamma)))))
    for _ in range(steps):
        mb.step(side / steps)
    inner = sp.interior(blk)
    rho = blk.Q.get()[inner][..., 0]
    error = np.abs(rho - start["rho"][inner])
    return error.max(), error.mean(), rho


# measured: max 1.150e-1 / 2.736e-2 (order 2.07, mean 2.19), 1.291e-1 / 1.717e-2 (2.91, 2.62), 6.168e-2 / 9.670e-3 (2.67, 3.09)
orders = [("kepaec", 1.9, 2.0, 2.736e-2), ("muscl-koren-rusanov", 2.4, 2.4, 1.717e-2), ("muscl-koren-ausmup", 2.4, 2.4, 9.670e-3)]


@pytest.mark.parametrize("name,atleast,mean_atleast,e80", orders, ids=[o[0] for o in orders])
def test_the_order_in_two_dimensions(backend, name, atleast, mean_atleast, e80):
    e40, e80got = vortex(backend, 41, **sp.schemes[name]), vortex(backend, 81, **sp.schemes[name])
    assert np.log2(e40[0] / e80got[0]) >= atleast
    assert np.log2(e40[1] / e80got[1]) >= mean_atleast
    assert e80got[0] <= 1.2 * e80


@pytest.mark.parametrize("name", ["kepaec", "muscl-koren-rusanov"])
def test_the_planes_agree_under_a_permutation(backend, name):
    xy = vortex(backend, 41, (0, 1), **sp.schemes[name])[2][:, :, 0]
    yz = vortex(backend, 41, (1, 2), **sp.schemes[name])[2][0, :, :]
    zx = vortex(backend, 41, (2, 0), **sp.schemes[name])[2][:, 0, :].T
    # measured 4.9e-12 and 7.5e-13: the reference energy -cp Tref is four hundred times the vortex's cv T, so each step's rounding is that much of the thermal energy
    assert np.allclose(xy, yz, rtol=1e-10, atol=0.0) and np.allclose(xy, zx, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("name", ["kepaec", "rusanov", "ausmup", "muscl-koren-rusanov", "muscl-koren-ausmup"])
def test_the_wave_errs_alike_along_every_axis(backend, name):
    along = [sp.wave_errors(backend, 41, axis=axis, **sp.schemes[name]) for axis in range(3)]
    # measured within 5e-13
    assert np.allclose(along[1], along[0], rtol=1e-11, atol=0.0)
    assert np.allclose(along[2], along[0], rtol=1e-11, atol=0.0)
