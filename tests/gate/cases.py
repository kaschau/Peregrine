import hashlib

import numpy as np
import peregrine as pg

from tests.gases import configure, primitives

cases = {
    "navstokes rk3": dict(physics="navier-stokes", gas="air", integrator="tvd-rk3"),
    "euler rk3": dict(physics="euler", gas="air", integrator="tvd-rk3"),
    # the Riemann solver with its low-Mach cutoff, on the cells as they are
    "euler rk3 ausmup": dict(
        physics="euler", gas="air", integrator="tvd-rk3", advflux=dict(primary="ausmup")
    ),
    # two cells a side through the Koren limiter: the second halo layer is read
    "euler rk3 muscl-koren-ausmup": dict(
        physics="euler",
        gas="air",
        integrator="tvd-rk3",
        advflux=dict(primary="ausmup", recon="muscl", limiter="koren"),
    ),
    "navstokes rk3 CH4_O2": dict(
        physics="navier-stokes", gas="CH4_O2", integrator="tvd-rk3"
    ),
    # binary diffusion: distinct diffusivities, so the species correction velocity is not zero as at one Lewis number
    "navstokes rk3 CH4_O2 binary": dict(
        physics="navier-stokes", gas="CH4_O2", integrator="tvd-rk3", diffusion="binary"
    ),
    # reacting: the production rates as the source, which nothing else here exercises
    "navstokes rk3 CH4_O2 explicit": dict(
        physics="navier-stokes", gas="CH4_O2", integrator="tvd-rk3", chemint="coupled"
    ),
    # sub-stepped: the bulk and dense passes, the list and its reset, under the gradient exchange
    "navstokes rk3 CH4_O2 sub-stepped": dict(
        physics="navier-stokes", gas="CH4_O2", integrator="tvd-rk3", chemint="sub-stepped"
    ),
}


def mesh():
    return pg.mesher.CubeMesher(
        mbdims=[2, 2, 1],
        blkdims=[8, 7, 6],
        lengths=[1, 1, 1],
        periodic=[True, False, True],
        bcnames={3: "walls", 4: "walls"},
    )


def build(
    backend,
    physics,
    gas,
    integrator,
    diffusion=None,
    chemint=None,
    advflux=None,
    ranks=(1, 1),
    grid=None,
):
    config = pg.Config()
    configure(config, gas, physics)
    if diffusion:
        config["mixture"]["diffusion"] = diffusion
    if chemint:
        config["mixture"]["chemistry"] = "finite-rate"
        config["chem-integration"]["kind"] = chemint
    for key, value in (advflux or {}).items():
        config["adv-flux"][key] = value
    config["time-integration"]["integrator"] = integrator
    config["simulation"]["dt"] = 1e-9
    # the j sides are walls: no-slip where the flow sticks, slip where it cannot
    wall = "no-slp-adia-wall" if physics == "navier-stokes" else "slp-adia-wall"
    config["bcvals"]["walls"] = {"kind": wall}
    mb = pg.get_solver(
        config, pg.readers.GridReader(grid, ranks=ranks) if grid else mesh(), backend(config)
    )
    prims = []
    for blk in mb.blocks:
        rng = np.random.default_rng(blk.nblki)
        q = primitives(mb, blk)
        q[..., 0] *= 1 + 0.05 * rng.random(q.shape[:3])
        q[..., 1:4] = 30 * rng.random(q.shape[:3] + (3,))
        q[..., 4] *= 1 + 0.1 * rng.random(q.shape[:3])
        if mb.sim.ne > 5:
            base = rng.random(mb.sim.ne - 5) / (mb.sim.ne - 5)
            q[..., 5:] = base * (1 + 0.01 * rng.random(q.shape[:3] + (mb.sim.ne - 5,)))
        prims.append(q)
    mb.set_prims(prims)
    # what each block started from, for the digest to refuse a state that stood still
    for blk in mb.blocks:
        blk.started = blk.Q.get()[mb.ng : -mb.ng, mb.ng : -mb.ng, mb.ng : -mb.ng].copy()
    return mb


# long enough that a one-ulp change anywhere reaches the hash
def step(mb, n=12):
    dt = mb.cfg["simulation"]["dt"]
    for _ in range(n):
        mb.step(dt)


def rhs(mb):
    mb.dt.set(mb.cfg["simulation"]["dt"])
    stage = mb.integ.stageregs(mb.regs)[0]
    mb.rhs(Q=stage.Q, dQ=stage.dQ)
    ng = mb.ng
    return {
        blk.nblki: getattr(blk, stage.dQ).get()[ng:-ng, ng:-ng, ng:-ng].copy()
        for blk in mb.blocks
    }


def block_digests(mb):
    ng = mb.ng
    digests = {}
    for blk in mb.blocks:
        Q = np.ascontiguousarray(blk.Q.get()[ng:-ng, ng:-ng, ng:-ng])
        # a NaN hashes as steadily as a number, so the state is refused first
        assert np.isfinite(Q).all(), f"block {blk.nblki}: the state is not finite"
        assert not np.array_equal(
            Q, blk.started
        ), f"block {blk.nblki}: the state never moved"
        digests[blk.nblki] = hashlib.sha256(Q.tobytes()).hexdigest()
    return digests


def digest(block_digests):
    h = hashlib.sha256()
    # in block order, so the digest is the same whatever rank holds which block
    for nblki in sorted(block_digests):
        h.update(block_digests[nblki].encode())
    return h.hexdigest()[:16]


def platform(mb):
    arch = mb.backend.compiler.arch
    return mb.backend.name + ("-" + arch.lower() if arch else "")
