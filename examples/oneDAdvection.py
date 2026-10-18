#!/usr/bin/env python
"""

Test case from

Preventing spurious pressure oscillations in split convective form discretization for compressible flows
https://doi.org/10.1016/j.jcp.2020.110060

Should reproduce results in Fig. 1 for the KEEP scheme (blue line)

    python examples/oneDAdvection.py <backend>

"""

from mpi4py import MPI  # noqa: F401

import sys

import peregrine as pg
import numpy as np
import matplotlib.pyplot as plt

# the ideal gas the case assumes, stated in full
air = {"Air": {"MW": 28.97, "cp0": 1000.0}}


def simulate(name, index="i"):
    config = pg.Config()
    config["simulation"]["simulator"] = "euler"
    config["mixture"]["species"] = air
    config["adv-flux"]["primary"] = "kepaec"
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    config.validate()
    backend = pg.get_backend(name, config)
    rot = {"i": 0, "j": 1, "k": 2}

    def rotate(li, index):
        return li[-rot[index] :] + li[: -rot[index]]

    nx = 41
    blkdims = rotate([nx, 2, 2], index)
    lengths = rotate([1, 0.1, 0.1], index)
    periodic = rotate([True, False, False], index)

    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1],
        blkdims=blkdims,
        lengths=lengths,
        periodic=periodic,
        bcnames=dict.fromkeys(range(1, 7), "walls"),
    )
    mb = pg.get_solver(config, mesh, backend)
    print(mb)

    blk = mb.blocks[0]
    ng = blk.ng
    R = 287.002507
    ccAxis = {"i": 0, "j": 1, "k": 2}
    uIndex = {"i": 1, "j": 2, "k": 3}
    # the primitive vector, p, u, v, w, T: unit pressure and speed, and the
    # temperature that gives the density wave
    q = np.zeros(blk.Q.shape[:3] + (mb.sim.ne,))
    q[:, :, :, 0] = 1.0
    q[:, :, :, uIndex[index]] = 1.0
    xc = blk.xcc.get()[..., ccAxis[index]]
    initial_rho = 2.0 + np.sin(2 * np.pi * xc)
    initial_T = 1.0 / (R * initial_rho)
    q[:, :, :, 4] = initial_T
    mb.set_prims([q])

    dt = 0.1 * 0.025
    tend = 11.0
    bar = pg.misc.Progress(tend)
    while mb.tme.get() < tend:
        if mb.nrt % 50 == 0:
            bar.at(mb.tme.get())

        mb.step(dt)

    data = mb.sim.iovars.data(blk, ["rho", "p", "u", "v", "w"])
    fig, ax1 = plt.subplots()
    ax1.set_title("1D Advection Results")
    ax1.set_xlabel(r"x")
    s_ = rotate(np.s_[ng:-ng, ng, ng], index)
    x = blk.xcc.get()[..., ccAxis[index]][s_]
    rho = data["rho"][s_]
    p = data["p"][s_]
    u = data["uvw"[ccAxis[index]]][s_]
    ax1.plot(x, rho, color="g", label="rho", linewidth=0.5)
    ax1.plot(x, p, color="r", label="p", linewidth=0.5)
    ax1.plot(x, u, color="k", label="u", linewidth=0.5)
    ax1.scatter(
        x,
        initial_rho[ng:-ng, ng:-ng, ng:-ng],
        marker="o",
        facecolor="w",
        edgecolor="b",
        label="exact",
        linewidth=0.5,
    )
    ax1.legend()
    plt.show()
    plt.close()
    backend.stop()


if __name__ == "__main__":
    try:
        simulate(sys.argv[1], "i")

    except Exception as e:
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
