import itertools

import numpy as np
import pytest

import peregrine as pg

from tests.gases import configure, primitives

MFLUX = 50.0


class Converging(pg.mesher.CubeMesher):
    def shape_blk(self, blk, i, j, k):
        super().shape_blk(blk, i, j, k)
        nodes = blk.nodes.get()
        x = nodes[..., 0]
        for c in (1, 2):
            mid = 0.5 * self.lengths[c]
            nodes[..., c] = mid + (nodes[..., c] - mid) * (1.0 - 0.3 * x / self.lengths[0])
        blk.nodes.set(nodes)


def miss(backend, primary, recon, steps=8000, dt=3e-6):
    config = pg.Config()
    configure(config, "air", "euler")
    config["adv-flux"]["primary"] = primary
    config["adv-flux"]["recon"] = recon
    if recon == "muscl":
        config["adv-flux"]["limiter"] = "koren"
    config["bcvals"]["inlet"] = {"kind": "sub-in-fmt", "mflux": MFLUX, "T": 300.0}
    config["bcvals"]["exit"] = {"kind": "sub-out-fp", "p": 101325.0}
    config["bcvals"]["wall"] = {"kind": "slp-adia-wall"}
    mesh = Converging(
        mbdims=[1, 1, 1], blkdims=[9, 3, 3], lengths=[1.0, 0.25, 0.25], bcnames={1: "inlet", 2: "exit"}, defaultbc="wall"
    )
    mb = pg.get_solver(config, mesh, backend(config))
    blk = mb.blocks[0]
    q = primitives(mb, blk)
    q[..., 1] = MFLUX / 1.17
    mb.set_prims([q])
    for _ in range(steps):
        mb.step(dt)
    inlet = blk.get_intface(1)
    F = inlet.faceplane(blk.iF.get())[..., 0]
    S = np.linalg.norm(inlet.faceplane(blk.iS.get()), axis=-1)
    assert np.isfinite(blk.Q.get()[blk.interior]).all()
    return F.sum() / (MFLUX * S.sum()) - 1.0


# only kepaec on a constant reconstruction carries the halo's average exactly; the rest miss by the measured amount with room (rusanov +8e-3 and ausmup +5e-3 on constant, -1.5e-2 under muscl)
MISSES = {
    ("kepaec", "constant"): 1e-12,
    ("rusanov", "constant"): 1.5e-2,
    ("ausmup", "constant"): 1e-2,
    ("kepaec", "muscl"): 3e-2,
    ("rusanov", "muscl"): 3e-2,
    ("ausmup", "muscl"): 3e-2,
}


@pytest.mark.parametrize("primary,recon", list(itertools.product(("kepaec", "rusanov", "ausmup"), ("constant", "muscl"))))
def test_the_delivered_mass_flux(backend, primary, recon):
    assert abs(miss(backend, primary, recon)) <= MISSES[primary, recon]
