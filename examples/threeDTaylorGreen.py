#!/usr/bin/env python
"""

Test case from

Preventing spurious pressure oscillations in split convective form discretization for compressible flows
https://doi.org/10.1016/j.jcp.2020.110060

Should reproduce results in Fig. 2 for the KEEP scheme (blue line)

    python examples/threeDTaylorGreen.py <backend>

"""

from mpi4py import MPI  # noqa: F401

import sys

import peregrine as pg
import numpy as np
import matplotlib.pyplot as plt

# the ideal gas the case assumes, stated in full
air = {"Air": {"MW": 28.97, "cp0": 1000.0}}


np.seterr(all="raise")


def simulate(name):
    config = pg.Config()
    config["simulation"]["simulator"] = "euler"
    config["mixture"]["species"] = air
    config.validate()
    backend = pg.get_backend(name, config)

    NE = 65
    NN = 65
    NX = 65
    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1],
        blkdims=[NE, NN, NX],
        lengths=[2 * np.pi for _ in range(3)],
        periodic=[True, True, True],
    )
    mb = pg.get_solver(config, mesh, backend)

    blk = mb.blocks[0]

    R = 287.002507
    cp = 1000.0
    cv = cp - R
    M0 = 0.4
    rho0 = 1.0
    gamma = cp / (cp - R)
    xc, yc, zc = (blk.xcc.get()[..., n] for n in range(3))
    # the primitive vector, p, u, v, w, T
    q = np.zeros(blk.Q.shape[:3] + (mb.sim.ne,))
    q[:, :, :, 0] = 1 / gamma + (rho0 * M0**2 / 16.0) * (
        np.cos(2 * xc) + np.cos(2 * yc)
    ) * (np.cos(2 * zc) + 2.0)
    q[:, :, :, 1] = M0 * np.sin(xc) * np.cos(yc) * np.cos(zc)
    q[:, :, :, 2] = -M0 * np.cos(xc) * np.sin(yc) * np.cos(zc)
    q[:, :, :, 4] = q[:, :, :, 0] / (R * rho0)

    mb.set_prims([q])

    dt = 0.1 * 2 * np.pi / 64
    ke = []
    e = []
    s = []
    t = []
    tend = 120.0 / M0
    bar = pg.misc.Progress(tend)
    while mb.tme.get() < tend:
        if mb.nrt % 50 == 0:
            bar.at(mb.tme.get())
            data = mb.sim.iovars.data(blk, ["rho", "p", "u", "v", "w", "T"])
            rho, p, u, v, w, T = (
                data[n][blk.interior] for n in "rho p u v w T".split()
            )

            rke = np.sum(0.5 * rho * (u**2 + v**2 + w**2))
            re = np.sum(rho * (T * cv))
            rS = np.sum(rho * np.log10(p * rho ** (-gamma)))

            ke.append(rke)
            e.append(re)
            s.append(rS)
            t.append(mb.tme.get() * M0)

        mb.step(dt)

    plt.plot(t, ke / ke[0])
    plt.ylim([0, 2.4])
    plt.title(r"$\rho k / (\rho k)_{0}$")
    plt.savefig("ke.png")
    plt.clf()
    plt.plot(t, (-(s - s[0])) / s[0])
    plt.ylim([-3e-2, 1e-2])
    plt.title(r"$\Delta(\rho s) / (\rho_0 s_0)$")
    plt.savefig("entropy.png")
    plt.clf()
    plt.plot(t, e / e[0])
    plt.savefig("e.png")
    plt.clf()
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
