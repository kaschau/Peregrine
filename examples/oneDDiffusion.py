#!/usr/bin/env -S python -m mpi4py
"""

Binary diffusion between two species of equal total mass inside box

    python examples/oneDDiffusion.py <backend>

"""

from mpi4py import MPI  # noqa: F401

import sys

import peregrine as pg
import numpy as np
import matplotlib.pyplot as plt


def simulate(name):
    config = pg.Config()
    config["simulation"]["simulator"] = "navier-stokes"
    config["mixture"]["species"] = ["O2", "N2"]
    config["mixture"]["eos"] = "tpg"
    config["mixture"]["trans"] = "kinetic-theory"
    config["mixture"]["T-range"] = (200.0, 1000.0)
    config["adv-flux"]["primary"] = "rusanov"
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    config.validate()
    backend = pg.get_backend(name, config)
    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1],
        blkdims=[41, 2, 2],
        lengths=[1, 0.01, 0.01],
        bcnames=dict.fromkeys(range(1, 7), "walls"),
    )
    mb = pg.get_solver(config, mesh, backend)

    blk = mb.blocks[0]
    ng = blk.ng
    # the primitive vector, p, u, v, w, T, Y(O2): one species each side of
    # the middle, the temperatures making the masses equal
    q = np.zeros(blk.Q.shape[:3] + (mb.sim.ne,))
    q[:, :, :, 0] = 101325.0
    MWA, MWB = (mb.sim.mixture.species[n]["MW"] for n in ("O2", "N2"))
    xc = blk.xcc.get()[..., 0]
    q[:, :, :, 4] = np.where(xc < 0.5, 300.0 * MWA / MWB, 300.0)
    q[:, :, :, 5] = np.where(xc < 0.5, 1.0, 0.0)
    mb.set_prims([q])

    dt = 1e-5
    nrt = 50000
    bar = pg.misc.Progress(nrt)
    while mb.nrt < nrt:
        mb.step(dt)
        if mb.nrt % 100 == 0:
            bar.at(mb.nrt)

    A = mb.sim.iovars.data(blk, ["O2"])["O2"]
    fig, ax1 = plt.subplots()
    ax1.set_title("1D Diffusion Results")
    ax1.set_xlabel(r"x")
    x = blk.xcc.get()[..., 0][ng:-ng, ng, ng]
    A = A[ng:-ng, ng, ng]
    B = 1.0 - A
    ax1.plot(x, A, marker="o", color="r", label="A", linewidth=1.0)
    ax1.plot(x, B, linestyle="--", color="k", label="B", linewidth=1.5)
    ax1.set_ylim([0.49, 0.51])
    ax1.legend()
    plt.show()
    plt.close()
    backend.stop()


if __name__ == "__main__":
    try:
        simulate(sys.argv[1])

    except Exception as e:
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
