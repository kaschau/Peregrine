import math

import numpy as np
import pytest

from peregrine.simulator import sem
from peregrine.simulator.sem.eddies import PdfPopulation, RandomPopulation
from peregrine.simulator.sem.flow import BoundaryLayer, Channel, Isotropic, Pipe
from peregrine.simulator.sem.models import Sigmas


def channel(retau=587.19, delta=0.05, viscosity=1.81e-5, zdim=0.1, **more):
    utau = retau * viscosity / delta
    return Channel(
        ydim=2.0 * delta,
        zdim=zdim,
        Uo=21.0 * utau,
        viscosity=viscosity,
        utau=utau,
        **more,
    )


def test_isotropic_models_at_scalar_y():
    flow = Isotropic(ydim=2.0, zdim=2.0, Uo=2.5, scale=0.3)
    assert flow.sigmas(0.0).shape == (3, 3) and np.all(flow.sigmas(0.0) == 0.3)
    assert flow.sigmas(np.zeros(4)).shape == (4, 3, 3)
    assert float(flow.profile(0.1)) == 2.5 and flow.profile([0.0, 0.5]).shape == (2,)
    assert np.array_equal(flow.stats(0.2), np.eye(3))
    assert flow.stats(np.zeros(5)).shape == (5, 3, 3)
    assert flow.delta is None and flow.periods() == {1: 2.0, 2: 2.0}


def test_populations_count_eddies():
    # c-eddy times the eddy box grown by sigma on every side, times the length convected in T, over the eddy volume; uniform sigmas make the count exact
    s, Uo, T = 0.5, 10.0, 1.0
    flow = Isotropic(ydim=2.0, zdim=2.0, Uo=Uo, scale=s)
    bounds = flow.sigmas.bounds(flow)

    def speed(d):
        return np.full(np.shape(d), Uo)

    for ceddy in (1.0, 2.0):
        want = round(ceddy * (2.0 + 2.0 * s) ** 2 * Uo * T / s**3)
        for population in (RandomPopulation(), PdfPopulation()):
            locs, speeds = population.arrivals(
                flow, bounds, ceddy, 10**7, 0.0, T, speed, np.random.default_rng(0)
            )
            assert len(locs) == want
            assert np.all(speeds == Uo)
    # with uniform sigmas the pdf population weights every cell alike and its reference volume is the eddy's, so both populations agree
    ys, zs, area = flow.grid(s)
    d = flow.wallcoord(ys, zs)
    pdf = PdfPopulation()
    w = pdf.weights(flow.sigmas, d)
    assert np.all(w == 1.0)
    refvol = pdf.refvol(flow.sigmas, bounds, d, w / (w.sum() * area), area)
    assert refvol == pytest.approx(s**3, rel=1e-12)


def test_pdf_uses_eddy_volume():
    # a component's eddy volume is the product over the three directions
    rng = np.random.default_rng(3)
    sigmas = rng.uniform(0.1, 1.0, size=(50, 3, 3))
    expected = np.min(sigmas[:, :, 0] * sigmas[:, :, 1] * sigmas[:, :, 2], axis=1)
    got = Sigmas.minvol(sigmas)
    assert np.allclose(got, expected, rtol=1e-14, atol=0.0)
    assert np.isclose(got.min(), np.min(np.prod(sigmas, axis=2)), rtol=1e-14)


def test_channel_mirrors_its_tables():
    # the data covers the bottom half; the top half is its mirror about the centreline, the shear stresses involving v changing sign
    flow = channel()
    delta, stats = flow.delta, flow.stats
    knots = stats.knots
    assert np.all(np.diff(knots) > 0.0)
    assert np.isclose(knots[-1], 2.0 * delta, rtol=0.0, atol=1e-15)
    lower = knots[knots < delta]
    bottom, top = stats(lower), stats(2.0 * delta - lower)
    sign = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]], dtype=np.float64)
    assert np.allclose(
        top, bottom * sign, rtol=1e-12, atol=1e-12 * np.abs(bottom).max()
    )
    assert np.allclose(flow.profile(lower), flow.profile(2.0 * delta - lower))


def test_moser_stresses():
    # Moser, Kim & Mansour (1999) at Re_tau 590: rms peaks u'+ 2.77 at y+ 15, v'+ 1.04 near 80, w'+ 1.37 near 40, -uv+ 0.86 near 45; the table resamples the profiles, so within 2 %
    flow = channel()
    utau, nu = flow.inputs["utau"], flow.inputs["viscosity"]
    stats = flow.stats
    lower = stats.knots <= flow.delta
    yplus = stats.knots[lower] * utau / nu
    R = stats.values[lower] / utau**2

    def peak(column):
        k = np.argmax(column)
        return column[k], yplus[k]

    assert stats(15.0 * nu / utau)[0, 0] / utau**2 == pytest.approx(2.77**2, rel=2e-2)
    value, at = peak(R[:, 0, 0])
    assert value == pytest.approx(2.77**2, rel=2e-2) and 12.0 < at < 18.0
    value, at = peak(R[:, 1, 1])
    assert value == pytest.approx(1.04**2, rel=2e-2) and 60.0 < at < 100.0
    value, at = peak(R[:, 2, 2])
    assert value == pytest.approx(1.37**2, rel=2e-2) and 30.0 < at < 55.0
    value, at = peak(-R[:, 0, 1])
    assert value == pytest.approx(0.86, rel=2e-2) and 35.0 < at < 55.0
    # the centreline: Ruv vanishes by symmetry and the profiles read (0.77, 0.62, 0.63)^2
    centre = stats(flow.delta) / utau**2
    assert centre[0, 1] == 0.0
    assert np.diag(centre) == pytest.approx([0.77**2, 0.62**2, 0.63**2], rel=3e-2)
    # v is the quietest component off the wall; the spanwise shears are DNS noise
    assert np.all(R[1:, 1, 1] < R[1:, 0, 0]) and np.all(R[1:, 1, 1] < R[1:, 2, 2])
    assert np.abs(R[:, 0, 2]).max() < 0.03 and np.abs(R[:, 1, 2]).max() < 0.03


def test_jarrin_scales():
    # Jarrin (2008) figs 6.11 and 7.4: length scales over delta from (0.0585, 0.025, 0.0702) at the wall to (0.4, 0.43, 0.467) at the centreline; the streamwise scale is the time scale convected at Uo
    flow = channel()
    delta, utau, Uo = flow.delta, flow.inputs["utau"], flow.inputs["Uo"]
    sigmas = flow.sigmas
    assert sigmas(0.0)[:, 1] / delta == pytest.approx([0.0585, 0.025, 0.0702], rel=1e-12)
    assert sigmas(delta)[:, 1] / delta == pytest.approx([0.4, 0.43, 0.467], rel=1e-12)
    assert np.array_equal(sigmas(0.3 * delta)[:, 1], sigmas(0.3 * delta)[:, 2])
    lower = sigmas.knots <= delta
    L = sigmas.values[lower, :, 1] / delta
    assert np.all(np.diff(L, axis=0) > 0.0)
    T = sigmas.values[lower, :, 0] / (delta / utau * Uo)
    y = sigmas.knots[lower] / delta
    assert y[np.argmax(T[:, 0])] == pytest.approx(0.0936, abs=1e-3)
    assert T[:, 0].max() == pytest.approx(0.1265, abs=1e-3)
    assert np.all(T[-1] < T[0])
    assert sigmas(0.0)[0, 0] == pytest.approx(0.077771008 * delta / utau * Uo, rel=1e-12)


def test_wall_profile_laws():
    # the sublayer is u+ = y+, the lower half rises, and at delta the wake law sits psi (1 - E(delta)) u_tau below Uo
    flow = channel()
    delta, utau = flow.delta, flow.inputs["utau"]
    nu, Uo = flow.inputs["viscosity"], flow.inputs["Uo"]
    profile = flow.profile
    assert profile(5.0 * nu / utau) == pytest.approx(5.0 * utau, rel=1e-12)
    assert profile(nu / utau) == pytest.approx(utau, rel=1e-12)
    n = profile.knots.shape[0]
    assert np.all(np.diff(profile.values[: n // 2]) >= 0.0)
    law = flow.section("profile")
    E = 0.5 * (1.0 + math.erf((1.0 - law["wake-mu"]) / (law["wake-sigma"] * math.sqrt(2.0))))
    assert profile(delta) == pytest.approx(Uo - law["psi"] * (1.0 - E) * utau, rel=1e-12)


def test_pipe_reads_the_channel_unmirrored():
    # the same data, delta the radius, from the wall to the axis only
    flow = channel()
    scalars = {k: flow.inputs[k] for k in ("Uo", "viscosity", "utau")}
    pipe = Pipe(R=flow.delta, **scalars)
    assert pipe.delta == flow.delta
    assert pipe.stats.knots.shape[0] * 2 - 1 == flow.stats.knots.shape[0]
    d = np.linspace(0.0, flow.delta, 7)
    for family in ("profile", "sigmas", "stats"):
        assert np.array_equal(getattr(pipe, family)(d), getattr(flow, family)(d))
    # a point at the wall sees the normal pointing to the axis
    (frame,) = pipe.localframe([flow.delta], [0.0])
    assert np.allclose(frame[0], [-1.0, 0.0]) and np.allclose(frame[1], [0.0, -1.0])


def test_eddy_count_guard():
    # linearBL sigmas shrink to y+ = 1 at the wall, so a random population would ask for ~1e15 eddies
    section = dict(
        flow=dict(
            type="boundary-layer",
            ydim=1.2,
            zdim=2.0,
            delta=1.0,
            Uo=10.0,
            viscosity=1e-5,
            utau=0.1,
        ),
        ensembletime=10.0,
        nframes=20,
        population="random",
        prefetch=False,
    )
    inlet = sem.SemInlet(section, [0.5], [1.0])
    with pytest.raises(ValueError, match="max-eddies"):
        inlet.current(0.0)


def wallFlow(cls, **scalars):
    if cls is Channel:
        return Channel(ydim=2.0, zdim=1.0, **scalars)
    return BoundaryLayer(ydim=1.2, zdim=1.0, delta=1.0, **scalars)


@pytest.mark.parametrize("cls", [Channel, BoundaryLayer])
def test_low_re_tau_raises(cls):
    # Re_tau = 50: the log layer's upper bound 3 sqrt(Re_tau) sits below y+ = 30
    with pytest.raises(ValueError, match="Re_tau"):
        wallFlow(cls, Uo=1.0, viscosity=1e-3, utau=0.05)


@pytest.mark.parametrize("cls", [Channel, BoundaryLayer])
def test_profile_knots_ascend(cls):
    knots = wallFlow(cls, Uo=20.0, viscosity=1e-3, utau=1.0).profile.knots
    assert np.all(np.diff(knots) >= 0.0)


def test_boundary_layer_eddies_end_at_delta():
    flow = BoundaryLayer(
        ydim=1.2, zdim=1.0, delta=1.0, Uo=10.0, viscosity=1e-5, utau=0.1
    )
    assert flow.eddytop == 1.0
    assert list(flow.has_eddies([0.5, 1.1], [0.5, 0.5])) == [True, False]
    # above the layer the profile holds its edge value
    assert flow.profile(1.15) == flow.profile(1.0)
    assert np.isclose(flow.profile(1.15), 10.0, rtol=1e-2)


def test_scalars_are_refused_by_name():
    with pytest.raises(TypeError, match="needs"):
        Channel(ydim=2.0, zdim=1.0, Uo=1.0)
    with pytest.raises(TypeError, match="not"):
        Isotropic(ydim=1.0, zdim=1.0, Uo=1.0, bogus=2.0)
    flow = channel()
    with pytest.raises(TypeError, match="delta"):
        flow.update({"delta": 1.0})
    with pytest.raises(KeyError, match="has no"):
        flow.section("bogus")
