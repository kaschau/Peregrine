import numpy as np

import peregrine as pg

from tests.gases import configure, primitives


# interior nodes jostled so no face normal lies exactly along an axis
class PerturbedCube(pg.mesher.CubeMesher):
    def __init__(self, rng, **kwargs):
        super().__init__(**kwargs)
        self.rng = rng

    def shape_blk(self, blk, i, j, k):
        super().shape_blk(blk, i, j, k)
        nodes = blk.nodes.get()
        inner = np.s_[1:-1, 1:-1, 1:-1]
        nodes[inner] += self.rng.uniform(-1, 1, nodes[inner].shape) * 0.02
        blk.nodes.set(nodes)


# six significant digits keep the value short in the kernel text it is baked into
def rounded(value):
    return float(f"{value:.6g}")


def smooth(rng, shape, ng):
    field = np.ones(shape)
    for axis, n in enumerate(shape):
        x = (np.arange(n) - ng + 0.5) / (n - 2 * ng)
        profile = np.cos(0.5 * np.pi * x + rng.uniform(0.0, 2.0 * np.pi))
        others = [a for a in range(3) if a != axis]
        field = field * np.expand_dims(profile, others)
    return field


def case(backend, bc, adv, gas, recon="muscl", simulator="euler", seed=0, entry=None):
    rng = np.random.default_rng(seed)
    config = pg.Config()
    config["adv-flux"]["primary"] = adv
    config["adv-flux"]["recon"] = recon
    if recon == "muscl":
        config["adv-flux"]["limiter"] = "koren"
    configure(config, gas, simulator)
    # every interface the same boundary, at random values the config's entry sets
    values = {
        "kind": bc,
        "p": rounded(rng.uniform(101325 * 0.9, 101325 * 1.1)),
        "u": rounded(rng.uniform(1, 1000)),
        "v": rounded(rng.uniform(1, 1000)),
        "w": rounded(rng.uniform(1, 1000)),
        "T": rounded(rng.uniform(300 * 0.9, 300 * 1.1)),
        "mflux": rounded(rng.uniform(1, 1000)),
    }
    values["pt"], values["Tt"] = values["p"], values["T"]
    spnames = pg.simulator.mixture.Mixture(config["mixture"]).spnames
    if len(spnames) > 1:
        Ybc = rng.uniform(0.0, 1.0, len(spnames))
        Ybc = Ybc / np.sum(Ybc)
        values.update(zip(spnames[:-1], map(rounded, Ybc)))
    values.update(entry or {})
    config["bcvals"]["outer"] = values

    mesh = PerturbedCube(
        rng,
        mbdims=[1, 1, 1],
        blkdims=[8, 6, 4],
        lengths=[1, 1, 1],
        bcnames=dict.fromkeys(range(1, 7), "outer"),
    )
    mb = pg.get_solver(config, mesh, backend(config))
    mixture = mb.sim.mixture
    blk = mb.blocks[0]

    q = primitives(mb, blk)
    shape, ng = q.shape[:3], blk.ng
    # p, T and Y vary smoothly so an extrapolated halo stays physical; the velocity varies cell by cell so every exit face sees flow both ways
    q[..., 0] = 101325.0 * (1.0 + 0.3 * smooth(rng, shape, ng))
    q[..., 1:4] = rng.uniform(-200.0, 200.0, shape + (3,))
    q[..., 4] = 1000.0 * (1.0 + 0.3 * smooth(rng, shape, ng))
    if mixture.ns > 1:
        base = rng.uniform(0.0, 1.0, mixture.ns - 1)
        Y = np.stack([1.0 + 0.5 * smooth(rng, shape, ng) for _ in base], axis=-1) * base
        # the carried species sum to nine tenths, so the last one is present too
        q[..., 5:] = 0.9 * Y / np.sum(Y, axis=-1, keepdims=True)
    mb.set_prims([q])
    return mb


def spoil(mb, blk):
    fields = list(mb.sim.statefields)
    held = blk.state.get().copy()
    held[..., fields.index("p")] *= 2.0
    held[..., fields.index("T")] += 100.0
    blk.state.set(held)


def assert_consistent(mb, blk):
    fields = list(mb.sim.statefields)
    rho, p, T, e = (fields.index(f) for f in ("rho", "p", "T", "e"))
    Q, held = blk.Q.get().copy(), blk.state.get().copy()
    mb.launch(mb.kernels["fromQ"](Q=mb.regs[0]), "all")
    again = blk.state.get().copy()
    blk.state.set(held)
    ng = blk.ng
    inner = np.s_[ng:-ng, ng:-ng]
    for interface in blk.intfaces:
        layers = zip(interface.halo(Q), interface.halo(held), interface.halo(again))
        for q, s, r in layers:
            q, s, r = q[inner], s[inner], r[inner]
            assert (q[..., 0] > 0.0).all() and (s[..., p] > 0.0).all()
            assert np.array_equal(q[..., 0], s[..., rho])
            assert np.array_equal(r[..., rho], s[..., rho])
            # tpg's three Newton steps re-land e, p and T within 2e-11
            for c in (e, p, T):
                assert np.allclose(r[..., c], s[..., c], rtol=1e-10, atol=0.0)
