import numpy as np
import pytest

import peregrine as pg
from peregrine.config import ConfigError

from tests.gases import configure, primitives
from peregrine.simulator import sem

dt = 1e-5
# the inlet is the low-i face at x = 0, so sem's x is the grid's
frame = {"origin": [0.0, 0.0, 0.0], "ey": [0.0, 1.0, 0.0], "ez": [0.0, 0.0, 1.0]}


# an ensemble sixteen steps long with nine frames, so a frame every two steps
def inletEntry(**changes):
    entry = {
        "kind": "sub-in-sem",
        "T": 300.0,
        "frame": dict(frame),
        "sem": {
            "flow": {
                "type": "channel",
                "y-dim": 0.1,
                "z-dim": 0.157,
                "periodic-z": True,
                "Uo": 4.52,
                "viscosity": 1.81e-5,
                "utau": 0.2126,
            },
            "ensemble-time": 16 * dt,
            "nframes": 9,
            "nblend": 3,
            "seed": 10101,
            "t0": 0.0,
        },
    }
    return merged(entry, changes)


def merged(base, changes):
    out = dict(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merged(out[key], value)
        else:
            out[key] = value
    return out


def case(backend, entry=None, integrator="tvd-rk3", mbdims=(1, 1, 1), simulator="euler", recon="constant"):
    config = pg.Config()
    configure(config, "air", simulator)
    config["time-integration"]["integrator"] = integrator
    config["adv-flux"]["recon"] = recon
    if recon == "muscl":
        config["adv-flux"]["limiter"] = "koren"
    config["simulation"]["dt"] = dt
    config["ics"]["u"] = 4.52
    config["bcvals"]["inlet"] = inletEntry() if entry is None else entry
    config["bcvals"]["exit"] = {
        "kind": "sub-out-fp",
        "p": 101325.0,
    }
    config["bcvals"]["wall"] = {"kind": "slp-adia-wall"}
    mesh = pg.mesher.CubeMesher(
        mbdims=list(mbdims),
        blkdims=[5, 9, 9],
        lengths=[0.2, 0.1, 0.157],
        bcnames={1: "inlet", 2: "exit"},
        defaultbc="wall",
    )
    mb = pg.get_solver(config, mesh, backend(config))
    return config, mb


def inletFaces(mb):
    return [(blk, face) for blk, face in mb.intfaces() if face.bcname == "inlet"]


def kindOf(mb):
    (kind,) = [k for k in mb.bckinds if k.name == "inlet"]
    return kind


def signalOf(mb):
    return kindOf(mb).signal


def halo(blk, face, array):
    ng = blk.ng
    return [h[ng:-ng, ng:-ng] for h in face.halo(array)]


# nearest the face first
def interior(blk, face, array):
    ng = blk.ng
    planes = np.moveaxis(array, face.axis, 0)[ng:-ng]
    inner = planes if face.is_low else planes[::-1]
    return [p[ng:-ng, ng:-ng] for p in inner]


def reference(inlet, R, t):
    return inlet.velocity(t) @ R.T


def haloVelocity(mb, blk, face, t):
    mb.tme.set(t)
    mb.apply_bcs("preadvflux")
    return halo(blk, face, primitives(mb, blk))[0][..., 1:4]


@pytest.mark.parametrize("recon, ng", [("constant", 1), ("muscl", 2)])
def test_halo_state_is_the_signal(backend, recon, ng):
    _, mb = case(backend, recon=recon)
    assert mb.ng == ng
    signal = signalOf(mb)
    blk, face = inletFaces(mb)[0]
    # the solver puts the boundaries at its start time as it is built
    assert signal.loaded == 0.0
    t = 3.7 * dt
    mb.tme.set(t)
    mb.apply_bcs("preadvflux")
    q = primitives(mb, blk)
    a, b = face.shape
    want = reference(signal.inlet, kindOf(mb).R, t).reshape(a, b, 3)
    # the mean flow enters along x, and the signal varies over the face
    assert np.mean(want[..., 0]) > 0.0 and np.std(want[..., 0]) > 0.0
    for layer in halo(blk, face, q):
        assert np.allclose(layer[..., 1:4], want, rtol=1e-10, atol=1e-12)
        assert np.allclose(layer[..., 4], 300.0)
    # the pressure is extrapolated, as the constant velocity inlet does it
    inner = interior(blk, face, q)
    for layer, deeper in zip(halo(blk, face, q), inner[1:]):
        assert np.allclose(layer[..., 0], 2.0 * inner[0][..., 0] - deeper[..., 0])


def test_coefficients_travel_once_per_interval(backend):
    _, mb = case(backend)
    signal = signalOf(mb)
    _, face = inletFaces(mb)[0]
    dtframe = 16 * dt / 8

    def boundariesAt(t):
        mb.tme.set(t)
        mb.apply_bcs("preadvflux")

    boundariesAt(0.3 * dtframe)
    first = signal.loaded
    boundariesAt(0.6 * dtframe)
    assert signal.loaded == first and face.tbreak.get()[0] == first
    # the time itself is the solver's clock, where every kernel reads it
    assert mb.tme.get() == pytest.approx(0.6 * dtframe)
    boundariesAt(1.2 * dtframe)
    assert signal.loaded == pytest.approx(dtframe) and signal.loaded != first
    assert face.tbreak.get()[0] == pytest.approx(dtframe)
    assert mb.tme.get() == pytest.approx(1.2 * dtframe)


# each scheme's stages by the fraction of the step their state is at, which is when its boundaries are put: at its readying for the next stage, the last one's at the next step's start
STAGES = [
    ("rk1", [1.0]),
    ("rk2", [1.0, 1.0]),
    ("tvd-rk3", [1.0, 0.5, 1.0]),
    ("rk34", [0.5, 1.0, 0.5, 1.0]),
    ("rk4", [0.5, 0.5, 1.0, 1.0]),
]


@pytest.mark.parametrize("integrator, fractions", STAGES)
def test_boundaries_are_updated_at_the_stage_times(backend, integrator, fractions):
    _, mb = case(backend, integrator=integrator)
    signal = signalOf(mb)
    seen, original = [], signal.update

    def recording():
        seen.append(mb.tme.get())
        original()

    signal.update = recording
    start = mb.tme.get()
    mb.step(dt)
    # the solution's readying stood from its setting, so the first step puts the boundaries at each stage's state but the last
    within = [start + f * dt for f in fractions[:-1]]
    assert seen == pytest.approx(within)
    assert mb.tme.get() == pytest.approx(start + dt)
    # every step after readies the solution first, the last stage's state at the time it is at
    seen.clear()
    start = mb.tme.get()
    mb.step(dt)
    assert seen == pytest.approx([start] + [start + f * dt for f in fractions[:-1]])


def test_a_viscous_inlet_is_the_same_signal(backend):
    _, mb = case(backend, simulator="navier-stokes")
    signal = signalOf(mb)
    t = 3.7 * dt
    mb.tme.set(t)
    mb.apply_bcs("preadvflux")
    for blk, face in inletFaces(mb):
        q = primitives(mb, blk)
        a, b = face.shape
        want = reference(signal.inlet, kindOf(mb).R, t).reshape(a, b, 3)
        for layer in halo(blk, face, q):
            assert np.allclose(layer[..., 1:4], want, rtol=1e-10, atol=1e-12)
            assert np.allclose(layer[..., 4], 300.0)
    blk = mb.blocks[0]
    before = blk.Q.get().copy()
    mb.apply_bcs("pregrads")
    assert np.array_equal(blk.Q.get(), before)
    rng = np.random.default_rng(17)
    blk.grads.set(rng.uniform(-1.0, 1.0, blk.grads.get().shape))
    mb.apply_bcs("postgrads")
    grads = blk.grads.get()
    for blk, face in inletFaces(mb):
        for layer, image in zip(halo(blk, face, grads), interior(blk, face, grads)):
            assert np.array_equal(layer, image)
    mb.step(dt)
    assert np.isfinite(blk.Q.get()[blk.interior]).all()


def test_a_chunk_boundary_passes_without_a_jump(backend):
    _, mb = case(backend)
    signal = signalOf(mb)
    blk, face = inletFaces(mb)[0]
    T = 16 * dt
    eps = 1e-12
    before = haloVelocity(mb, blk, face, T - eps)
    after = haloVelocity(mb, blk, face, T + eps)
    # the next chunk's first interval starts at the boundary knot
    assert signal.loaded == pytest.approx(T)
    assert np.allclose(before, after, atol=1e-7)


def test_faces_of_one_name_share_one_signal(backend):
    _, mb = case(backend, mbdims=(1, 2, 1))
    kind = kindOf(mb)
    faces = inletFaces(mb)
    assert len(faces) == 2 and len(kind.interfaces) == 2
    t = 2.3 * dt
    mb.tme.set(t)
    mb.apply_bcs("preadvflux")
    for (blk, face), y, z in zip(faces, kind.y, kind.z):
        # a signal over this face's points alone gives the same values
        with sem.SemInlet(inletEntry()["sem"], y.ravel(), z.ravel()) as own:
            a, b = face.shape
            want = reference(own, kind.R, t).reshape(a, b, 3)
        got = halo(blk, face, primitives(mb, blk))[0][..., 1:4]
        assert np.allclose(got, want, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("integrator", ["tvd-rk3"])
def test_restart_continues_bitwise(backend, tmp_path, integrator):
    config, mb = case(backend, integrator=integrator)
    pg.writers.GridWriter(mb, str(tmp_path / "g.h5"), precision="double").write(mb)
    for _ in range(3):
        mb.step(dt)
    writer = pg.writers.RestartWriter(
        mb,
        mb.sim.iovars,
        str(tmp_path / "q.{n:08d}.h5"),
        "g.h5",
        "double",
        cfg=config,
    )
    writer.write(mb, mb.nrt, mb.tme.get())
    for _ in range(2):
        mb.step(dt)
    expect = [blk.Q.get()[blk.interior].copy() for blk in mb.blocks]

    reader = pg.readers.RestartReader(str(tmp_path / "q.00000003.h5"))
    again = pg.get_solver(reader.cfg, pg.readers.GridReader(reader.grid), backend(reader.cfg), reader)
    assert again.tme.get() == pytest.approx(3 * dt)
    for _ in range(2):
        again.step(dt)
    for (_, face), (_, faceAgain) in zip(inletFaces(mb), inletFaces(again)):
        assert np.array_equal(faceAgain.semcoeffs.get(), face.semcoeffs.get())
        assert np.array_equal(faceAgain.tbreak.get(), face.tbreak.get())
    assert again.tme.get() == mb.tme.get()
    # a few ulps of rho E from the primitives round trip reach every momentum component through the pressure gradient, so the momentum is judged on the momentum's scale, not each component's own
    for a, b in zip([blk.Q.get()[blk.interior].copy() for blk in again.blocks], expect):
        scale = np.abs(b).max(axis=(0, 1, 2))
        scale[1:4] = scale[1:4].max()
        assert (np.abs(a - b).max(axis=(0, 1, 2)) / scale).max() < 5e-14


@pytest.mark.parametrize(
    "changes, error, match",
    [
        # what sem or Python refuse on their own
        ({"frame": {"origin": "auto"}}, ValueError, "auto"),
        ({"sem": {"overlapFrames": 8}}, TypeError, "overlapFrames"),
        ({"sem": {"flow": {"type": "sphere"}}}, KeyError, "sphere"),
        ({"sem": {"t0": 1.0}}, ValueError, "outside"),
        # what would otherwise be silently wrong
        (
            {"frame": {"ey": [0.0, 0.0, 1.0], "ez": [0.0, 1.0, 0.0]}},
            ConfigError,
            "out of the block",
        ),
        (
            {"frame": {"ey": [0.0, 1.0, 0.0], "ez": [0.0, 1.0, 1.0]}},
            ConfigError,
            "orthogonal",
        ),
    ],
)
def test_what_the_entry_gets_wrong_is_refused(backend, changes, error, match):
    with pytest.raises(error, match=match):
        case(backend, entry=inletEntry(**changes))


def test_a_frame_without_an_origin_is_refused(backend):
    entry = inletEntry()
    del entry["frame"]["origin"]
    with pytest.raises(KeyError, match="origin"):
        case(backend, entry=entry)
