import numpy as np
import pytest

from peregrine.simulator import sem
from peregrine.simulator.sem.models import retau, yplus1

from tests.sem.gate import channel


def firstChunk(flow, tme, ys, zs, nframes, **kwargs):
    section = {
        "flow": flow,
        "ensemble-time": tme,
        "nframes": nframes,
        "nblend": 0,
        "prefetch": False,
        **kwargs,
    }
    with sem.SemInlet(section, ys, zs) as inlet:
        return inlet, inlet.current(inlet.t0)


def test_flow_scalars():
    flow = dict(Uo=1.0, viscosity=1e-5, delta=1.0, utau=0.05)
    assert yplus1(flow) == 1e-5 / 0.05
    assert retau(flow) == 0.05 * 1.0 / 1e-5
    assert yplus1(dict(Uo=1.0, viscosity=1e-5)) is None


def test_channel_frames():
    flow, tme = channel(2.0)
    nframes, delta = 60, 0.5 * flow["y-dim"]
    ys = np.linspace(0.05 * delta, 1.95 * delta, 6)
    zs = np.ones(ys.shape[0]) * flow["z-dim"] / 2.0
    inlet, chunk = firstChunk(
        flow, tme, ys, zs, nframes, population="random", interp=False
    )
    assert chunk.frames.shape == (nframes, ys.shape[0], 3)
    assert chunk.c.shape == (4, nframes - 1, ys.shape[0], 3)
    assert np.all(np.isfinite(chunk.frames))
    # the ensemble is whitened and given the Cholesky factor of Rij, so the sample stresses of the frames are the target
    prime = chunk.frames.copy()
    prime[:, :, 0] -= inlet.flow.profile(inlet.d)
    Rij = inlet.flow.stats(inlet.d)
    cov = np.einsum("tpi,tpj->pij", prime, prime) / nframes
    assert np.allclose(cov, Rij, rtol=1e-8, atol=1e-12 * np.abs(Rij).max())
    assert np.allclose(np.mean(prime[:, :, 0], axis=0), 0.0, atol=1e-8)


@pytest.mark.parametrize(
    "kind, shape, r",
    [("pipe", {"R": 0.05}, 0.025), ("annulus", {"Ri": 0.05, "Ro": 0.15}, 0.1)],
)
def test_round_frames(kind, shape, r):
    # a round section's point has its own wall frame: the imposed stresses in plane components are Rij turned by M = [[1, 0, 0], [0, n, t]] on (y, z)
    flow, tme = channel(0.5)
    scalars = {k: flow[k] for k in ("Uo", "viscosity", "utau")}
    angles = np.deg2rad([0.0, 45.0, 90.0, 135.0, 200.0, 300.0])
    ys, zs = r * np.cos(angles), r * np.sin(angles)
    nframes = 40
    inlet, chunk = firstChunk(
        {"type": kind, **shape, **scalars}, tme, ys, zs, nframes, population="pdf", interp=False
    )
    prime = chunk.frames.copy()
    prime[:, :, 0] -= inlet.flow.profile(inlet.d)
    Rij = inlet.flow.stats(inlet.d)
    cov = np.einsum("tpi,tpj->pij", prime, prime) / nframes
    n, t = inlet.ptframes[:, 0], inlet.ptframes[:, 1]
    for p in range(ys.shape[0]):
        M = np.array([[1.0, 0.0, 0.0], [0.0, n[p, 0], t[p, 0]], [0.0, n[p, 1], t[p, 1]]])
        turned = M @ Rij[p] @ M.T
        # measured 6e-14 of max |Rij|
        assert np.allclose(cov[p], turned, rtol=1e-10, atol=1e-12 * np.abs(Rij[p]).max())
    # off the axes the turning is not the identity
    assert not np.allclose(cov, Rij, rtol=1e-3)


def centreline(flow):
    return np.array([0.5 * flow["y-dim"]]), np.array([0.5 * flow["z-dim"]])


def test_coefficients_reproduce_the_frames():
    flow, tme = channel(0.5)
    ys, zs = centreline(flow)
    _, chunk = firstChunk(flow, tme, ys, zs, 20, interp=False)
    for i, t in enumerate(chunk.tbreaks[:-1]):
        tbreak, slab = chunk.coeffs(t)
        assert tbreak == t
        assert np.allclose(slab[3], chunk.frames[i])
    with pytest.raises(ValueError, match="outside"):
        chunk.coeffs(chunk.tend + 1.0)


def test_local_convection_wall_line_finite():
    # the mean profile is zero at the wall; local convection must not divide by it
    flow, tme = channel(0.5)
    ys = np.array([0.0, 0.5 * flow["y-dim"]])
    zs = np.full(2, 0.5 * flow["z-dim"])
    _, chunk = firstChunk(flow, tme, ys, zs, 20, convect="local")
    assert np.all(np.isfinite(chunk.frames))


def test_mismatched_shapes_raise():
    flow, tme = channel(0.5)
    section = dict(flow=flow, ensembletime=tme, nframes=20, prefetch=False)
    with pytest.raises(ValueError, match="same shape"):
        sem.SemInlet(section, [0.5 * flow["y-dim"]], [0.0, 0.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"convect": "bogus"},
        {"population": "bogus"},
        {"flow": {"type": "sphere"}},
    ],
)
def test_unknown_names_are_refused(kwargs):
    flow, tme = channel(0.5)
    section = dict(flow=flow, ensembletime=tme, nframes=20, prefetch=False)
    section.update(kwargs)
    with pytest.raises(KeyError):
        sem.SemInlet(section, *centreline(flow))


def test_options_not_taken_are_refused():
    # one eddy shape and one normalization, not chosen by name; delta is the channel's half-height, not an input
    flow, tme = channel(0.5)
    for kwargs in ({"shape": "blob"}, {"normalization": "exact"}):
        section = dict(flow=flow, ensembletime=tme, nframes=20, prefetch=False)
        section.update(kwargs)
        with pytest.raises(TypeError):
            sem.SemInlet(section, *centreline(flow))
    section = dict(
        flow={**flow, "delta": 0.05}, ensembletime=tme, nframes=20, prefetch=False
    )
    with pytest.raises(TypeError, match="delta"):
        sem.SemInlet(section, *centreline(flow))
