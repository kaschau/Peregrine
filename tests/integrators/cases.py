import numpy as np

import peregrine as pg

from tests.gases import configure, primitives

schemes = ("rk1", "rk2", "tvd-rk3", "rk34", "rk4")

# every carried species of CH4_O2_FFCMY positive, CO2 taking the rest
mixture = {"CH4": 0.2, "O2": 0.6}
mixture.update({s: 0.01 for s in ("H2", "H", "O", "OH", "HO2", "H2O", "CH3", "CO", "CH2O")})


def box(
    physics="euler",
    gas="air",
    integrator="tvd-rk3",
    controller="fixed",
    dt=1e-6,
    mbdims=(1, 1, 1),
    dims=(6, 6, 6),
    lengths=(1.0, 1.0, 1.0),
    periodic=(True, True, True),
    names=None,
    defaultbc=None,
    **simulation,
):
    config = pg.Config()
    configure(config, gas, physics)
    config["time-integration"]["integrator"] = integrator
    config["simulation"]["controller"] = controller
    config["simulation"]["dt"] = dt
    for key, value in simulation.items():
        config["simulation"][key] = value
    if gas == "CH4_O2":
        config["ics"]["Y"] = dict(mixture)
    mesh = pg.mesher.CubeMesher(
        mbdims=list(mbdims),
        blkdims=list(dims),
        lengths=list(lengths),
        periodic=list(periodic),
        bcnames=names or {},
        defaultbc=defaultbc,
    )
    return config, mesh


def waves(rng, x, lengths, n=3):
    out = np.zeros(x.shape[:-1])
    for _ in range(n):
        k = rng.integers(1, 3, size=3)
        phase = rng.random() * 2 * np.pi
        out += rng.random() * np.sin(2 * np.pi * (x * k / lengths).sum(axis=-1) + phase)
    return out / n


def perturb(mb, rng, lengths=(1.0, 1.0, 1.0)):
    lengths = np.asarray(lengths, float)
    prims = []
    for blk in mb.blocks:
        x = blk.xcc.get()
        q = primitives(mb, blk)
        q[..., 0] *= 1 + 0.05 * waves(rng, x, lengths)
        for c in range(1, 4):
            q[..., c] = 30.0 * waves(rng, x, lengths)
        q[..., 4] *= 1 + 0.1 * waves(rng, x, lengths)
        for n in range(5, q.shape[-1]):
            q[..., n] *= 1 + 0.1 * waves(rng, x, lengths)
        prims.append(q)
    mb.set_prims(prims)


def totals(mb, register="Q"):
    total = scale = 0.0
    for blk in mb.blocks:
        volume = 1.0 / blk.detJinv.get()[blk.interior][..., None]
        q = getattr(blk, register).get()[blk.interior] * volume
        total = total + q.sum(axis=(0, 1, 2))
        scale = scale + np.abs(q).sum(axis=(0, 1, 2))
    return total, scale


class TurnedCubeMesher(pg.mesher.CubeMesher):
    def __init__(self, rot, **kwargs):
        super().__init__(**kwargs)
        self.rot = np.asarray(rot, float)

    def shape_blk(self, blk, i, j, k):
        super().shape_blk(blk, i, j, k)
        nodes = blk.nodes.get()
        nodes[blk.interior] = nodes[blk.interior] @ self.rot.T
        blk.nodes.set(nodes)
