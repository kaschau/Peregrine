import json
import sys
from pathlib import Path

import numpy as np

from peregrine.simulator import sem
from peregrine.simulator.sem.march import march

# local to the machine and not committed; re-record only for an intentional change in behaviour
REFERENCES = Path(__file__).parent / "references"

# leading eddies stored verbatim; the rest are pinned by column sums
EDDY_HEAD = 512


def channel(boxTurns):
    retau, delta, viscosity = 587.19, 0.05, 1.81e-5
    utau = retau * viscosity / delta
    Uo = 2.1263e01 * utau
    flow = {
        "type": "channel",
        "y-dim": 2.0 * delta,
        "z-dim": np.pi * delta,
        "periodic-z": True,
        "Uo": Uo,
        "viscosity": viscosity,
        "utau": utau,
    }
    return flow, boxTurns * 2 * np.pi * delta / Uo


def boundaryLayer():
    delta = 1.0
    flow = {
        "type": "boundary-layer",
        "y-dim": 1.2 * delta,
        "z-dim": 2.0 * delta,
        "periodic-z": True,
        "Uo": 10.0,
        "viscosity": 1e-5,
        "utau": 0.1,
        "delta": delta,
    }
    return flow, 2.0


def isotropic():
    width = 2.0 * np.pi
    flow = {
        "type": "isotropic",
        "y-dim": width,
        "z-dim": width,
        "Uo": 10.0,
        "viscosity": 1e-5,
        "scale": 0.5,
    }
    return flow, 3.0


def channelLine(flow, ylo, yhi, n):
    delta = 0.5 * flow["y-dim"]
    return np.linspace(ylo * delta, yhi * delta, n), np.full(n, flow["z-dim"] / 2.0)


def caseC1():
    flow, tme = channel(0.5)
    ys, zs = channelLine(flow, 0.05, 1.95, 6)
    return (
        flow,
        ys,
        zs,
        tme,
        dict(nframes=60, population="random", interp=False),
    )


def caseC2():
    flow, tme = channel(0.5)
    ys, zs = channelLine(flow, 0.5, 1.5, 2)
    return (
        flow,
        ys,
        zs,
        tme,
        dict(nframes=20, population="random", interp=False),
    )


def caseC3():
    flow, tme = channel(0.5)
    ys, zs = channelLine(flow, 0.01, 1.99, 6)
    return (
        flow,
        ys,
        zs,
        tme,
        dict(nframes=40, population="pdf", convect="local", interp=False),
    )


def caseC4():
    flow, tme = boundaryLayer()
    ys = np.linspace(0.001 * flow["y-dim"], 0.999 * flow["y-dim"], 5)
    zs = np.full(ys.shape[0], flow["z-dim"] / 2.0)
    return flow, ys, zs, tme, dict(nframes=40, population="pdf", interp=True)


def caseC5():
    flow, tme = isotropic()
    ys, zs = np.array([np.pi]), np.array([np.pi])
    return (
        flow,
        ys,
        zs,
        tme,
        dict(nframes=500, population="random", ceddy=2.0, interp=False),
    )


def roundSection(kind, **shape):
    flow, tme = channel(0.5)
    scalars = {k: flow[k] for k in ("Uo", "viscosity", "utau")}
    return {"type": kind, **shape, **scalars}, tme


def ring(r, n=6):
    # off the axes, so every frame turns
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + 0.1
    return r * np.cos(angles), r * np.sin(angles)


def caseC6():
    delta = 0.05
    flow, tme = roundSection("pipe", R=delta)
    ys, zs = ring(0.5 * delta)
    return flow, ys, zs, tme, dict(nframes=40, population="pdf", interp=False)


def caseC7():
    delta = 0.05
    flow, tme = roundSection("annulus", Ri=delta, Ro=3.0 * delta)
    ys, zs = ring(2.0 * delta)
    return flow, ys, zs, tme, dict(nframes=40, population="pdf", interp=False)


CASES = {
    "C1_channelRandom": caseC1,
    "C2_channelRandomShort": caseC2,
    "C3_channelPdfLocal": caseC3,
    "C4_boundaryLayerPdfInterpolate": caseC4,
    "C5_isotropic": caseC5,
    "C6_pipePdf": caseC6,
    "C7_annulusPdf": caseC7,
}


def run(case):
    flow, ys, zs, tme, kwargs = CASES[case]()
    section = {
        "flow": flow,
        "ensemble-time": tme,
        "nblend": 0,
        "prefetch": False,
        **kwargs,
    }
    inlet = sem.SemInlet(section, ys, zs)
    chunk = inlet.current(inlet.t0)
    eddies = inlet.cache.eddies(inlet.cache.live)
    # the sums before the statistics are imposed, which the whitening would otherwise hide a scale error in
    raw, _ = march(
        inlet.ys,
        inlet.zs,
        eddies,
        chunk.tbreaks - inlet.t0,
        inlet.has_eddies,
        inlet.ptframes,
    )
    up = chunk.frames[:, :, 0] - inlet.flow.profile(inlet.d)
    return {
        "params": np.array(json.dumps(dict(totalTime=tme, **kwargs))),
        "ys": ys,
        "zs": zs,
        "neddy": np.array(eddies.neddy),
        "eddyLocsHead": eddies.locs[:EDDY_HEAD],
        "epsHead": eddies.eps[:EDDY_HEAD],
        "sigmasHead": eddies.sigmas[:EDDY_HEAD],
        "eddySum": np.concatenate(
            (
                eddies.locs.sum(axis=0),
                eddies.eps.sum(axis=0),
                eddies.sigmas.sum(axis=0).ravel(),
            )
        ),
        "raw": raw,
        "up": up,
        "vp": chunk.frames[:, :, 1],
        "wp": chunk.frames[:, :, 2],
    }


def write(cases):
    REFERENCES.mkdir(exist_ok=True)
    for case in cases:
        result = run(case)
        np.savez(REFERENCES / f"{case}.npz", **result)
        print(
            f"{case}.npz: neddy={int(result['neddy'])}, up.shape={result['up'].shape}"
        )


def diff(cases):
    for case in cases:
        ref = np.load(REFERENCES / f"{case}.npz")
        got = run(case)
        same = all(
            np.array_equal(got[k], ref[k])
            for k in ("eddyLocsHead", "epsHead", "sigmasHead", "eddySum")
        )
        print(
            f"{case}: neddy {int(ref['neddy'])} -> {int(got['neddy'])}, eddies {'identical' if same else 'changed'}"
        )
        if not same:
            for k in ("eddyLocsHead", "eddySum"):
                print(f"    {k}: max |change| {np.max(np.abs(got[k] - ref[k])):.2e}")
        for key in ("raw", "up", "vp", "wp"):
            scale = np.max(np.abs(ref[key]))
            perPoint = np.max(np.abs(got[key] - ref[key]), axis=0) / scale
            print(f"    {key}: max change {perPoint.max():.2e} of max|{key}|")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] not in ("--write", "--diff"):
        raise SystemExit("usage: python -m tests.sem.gate --write|--diff [CASE ...]")
    cases = args[1:] or sorted(CASES)
    (write if args[0] == "--write" else diff)(cases)
