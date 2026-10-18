#!/usr/bin/env python
"""

A time trial test for testing and stuff.

Writes the case: a periodic box of nbi^3 blocks of ni^3 nodes of still air,
as g.h5, q.00000000.h5 and peregrine.yaml. The trial is the executable on
as many ranks as the grid is partitioned for, and its timing is what the
report plugin prints at the end:

    peregrine partition g.h5 -ranks 64 -ranks_per_node 4
    mpiexec -n 64 peregrine run peregrine.yaml g.h5 -b <backend>

"""

import peregrine as pg
import numpy as np

# a calorically perfect air, stated in full: the library carries no such species
air = {
    "Air": {
        "MW": 28.97,
        "cp0": 1002.838449439523,
        "mu0": 1.8591191080521142e-05,
        "kappa0": 0.02625394405190068,
    }
}


np.seterr(all="raise")


def simulate():
    config = pg.Config()
    config["simulation"]["simulator"] = "navier-stokes"
    config["mixture"]["species"] = air
    config["mixture"]["trans"] = "constant"

    config["simulation"]["niter"] = 100

    config["time-integration"]["integrator"] = "tvd-rk3"
    config["simulation"]["controller"] = "cfl"
    # the run prints nothing but its plugins: the report's closing lines are the trial's wall time and seconds per step per cell
    config["plugins"]["report"] = {"niter-out": config["simulation"]["niter"]}
    config.validate()

    ni = 30
    nbi = 10
    # a single species: the primitive variables are p, u, v, w, T
    iovars = pg.simulator.get_simulator(config).iovars.minimal
    mb = pg.multiblock.Restart(len(iovars.privars))
    pg.mesher.CubeMesher(
        mbdims=[nbi, nbi, nbi],
        blkdims=[ni, ni, ni],
        lengths=[1, 1, 1],
        periodic=[True, True, True],
    ).fill(mb)

    for blk in mb.blocks:
        q = blk.q.get()
        q[:, :, :, 0] = 101325.0
        q[:, :, :, 4] = 300.0
        blk.q.set(q)

    pg.writers.GridWriter(mb, "g.h5").write(mb)
    # the result carries the config, so the trial can also start from it: peregrine run -r q.00000000.h5 -b <backend>
    pg.writers.RestartWriter(mb, iovars, "q.{n:08d}.h5", "g.h5", cfg=config).write(mb, 0, 0.0)
    pg.write_cfg(config, "peregrine.yaml")


if __name__ == "__main__":
    try:
        simulate()

    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
