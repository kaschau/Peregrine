import numpy as np
import pytest

import peregrine as pg
from peregrine.plugins import BaseSimulatorPlugin

from tests.gases import air, configure

MULT = 20.0
MU0 = air["Air"]["mu0"]

# lines over the unit square: along x, along y, and oblique to both
LINES = {
    "x": ([0.25, 0.0, 0.0], [0.75, 0.0, 0.0]),
    "y": ([0.0, 0.25, 0.0], [0.0, 0.75, 0.0]),
    "oblique": ([0.25, 0.25, 0.0], [0.75, 0.75, 0.0]),
}


def square(config):
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    return pg.mesher.CubeMesher(
        mbdims=[1, 1, 1],
        blkdims=[9, 9, 3],
        lengths=[1.0, 1.0, 0.25],
        bcnames=dict.fromkeys(range(1, 7), "walls"),
    )


def profile(x, origin, ending):
    origin, ending = np.array(origin), np.array(ending)
    length = np.linalg.norm(ending - origin)
    normal = (ending - origin) / length
    along = (x @ normal - origin @ normal) / length
    return 1.0 + np.clip(along * (MULT - 1.0), 0.0, MULT - 1.0)


@pytest.mark.parametrize("line", LINES)
def test_the_viscosity_ramps_along_the_line(backend, line):
    origin, ending = LINES[line]
    config = pg.Config()
    configure(config, "air")
    config["plugins"]["viscsponge"] = {"origin": origin, "ending": ending, "multiplier": MULT}
    mb = pg.get_solver(config, square(config), backend(config))
    blk = mb.blocks[0]
    x = blk.xcc.get()[blk.interior]
    mu = blk.trstate.get()[blk.interior][..., 0]
    expected = MU0 * profile(x, origin, ending)
    # the box has cells before the origin, left alone, and past the ending, held at the multiplier
    assert (expected == MU0).any() and (expected == MULT * MU0).any()
    assert np.allclose(mu, expected, rtol=1e-12, atol=0.0)
    # the transport kernel remakes the viscosity ahead of the sponge in the same slot, so another readying leaves it
    mb.postrhs()
    assert np.array_equal(blk.trstate.get()[blk.interior][..., 0], mu)
    # and the sponge follows the transport wherever a register is readied: every stage's prerhs
    (plugin,) = mb.sim.plugins.values()
    readying = [g for (slot, *_), g in mb.graphs.items() if slot == "prerhs"]
    assert len(readying) > 1 and all(plugin.kernel in g.kernels for g in readying)


class Nowhere(BaseSimulatorPlugin):
    name = "nowhere"
    simulators = ".*"

    def launches(self, provider):
        return {"nowhere": []}


class Beginner(BaseSimulatorPlugin):
    name = "beginner"
    simulators = ".*"

    def launches(self, provider):
        return {"begin_dQ": []}


@pytest.mark.parametrize("plugin", ["nowhere", "beginner"])
def test_a_plugin_may_only_follow_a_slot(backend, plugin):
    # the begin among the slots refused: the apply adds to the derivative only what the physics begins
    config = pg.Config()
    configure(config, "air")
    config["plugins"][plugin] = {}
    with pytest.raises(ValueError, match="no slot"):
        pg.get_solver(config, square(config), backend(config))
