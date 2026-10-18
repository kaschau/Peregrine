#!/usr/bin/env python
"""

Test case from

High-order accurate kinetic-energy and entropy preserving (KEEP) schemes on curvilinear grids
https://doi.org/10.1016/j.jcp.2021.110482

Should reproduce figure 3 depending on resolution setting.

    python examples/twoDEulerVortex.py <backend>

"""

from mpi4py import MPI  # noqa: F401
import sys
import peregrine as pg
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

# the ideal gas the case assumes, stated in full
air = {"Air": {"MW": 28.97, "cp0": 1000.0}}


np.seterr(all="raise")


class SkewedSheet(pg.mesher.CubeMesher):
    """A sheet periodic in x and y whose nodes are moved off their lines by
    a product of sines, so no cell is a box."""

    def shape_blk(self, blk, i, j, k):
        super().shape_blk(blk, i, j, k)
        NE, NN, _ = self.blkdims
        Lx = Ly = 12.0
        delX, delY = Lx / (NE - 1), Ly / (NN - 1)
        Ax = Ay = 1.0
        xMin = yMin = -6.0
        lamX = lamY = 4.0
        kappa = 0.25
        E, N = np.meshgrid(np.arange(NE), np.arange(NN), indexing="ij")
        skew = np.sin(2 * np.pi * kappa)
        nodes = blk.nodes.get()
        skewed = nodes[blk.interior]
        skewed[..., 0] = (
            xMin + delX * (E + Ax * skew * np.sin(lamX * np.pi * N * delY / Ly))
        )[..., None]
        skewed[..., 1] = (
            yMin + delY * (N + Ay * skew * np.sin(lamY * np.pi * E * delX / Lx))
        )[..., None]
        blk.nodes.set(nodes)


def simulate(name):
    config = pg.Config()
    config["simulation"]["simulator"] = "euler"
    config["mixture"]["species"] = air
    config["bcvals"]["sides"] = {"kind": "slp-adia-wall"}
    config.validate()
    backend = pg.get_backend(name, config)

    NE = NN = 41
    Lx = 12.0
    mesh = SkewedSheet(
        mbdims=[1, 1, 1],
        blkdims=[NE, NN, 2],
        lengths=[12, 12, 0.01],
        periodic=[True, True, False],
        bcnames={5: "sides", 6: "sides"},
    )
    mb = pg.get_solver(config, mesh, backend)
    blk = mb.blocks[0]
    ng = blk.ng
    x0 = y0 = 0.0
    print(mb)

    Rc = 1.0
    rhoInf = 1.0
    MInf = 0.1
    pInf = 101325.0
    R = 287.002507
    cp = 1000.0
    cv = cp - R
    gamma = cp / cv
    TInf = pInf / (R * rhoInf)
    aInf = np.sqrt(gamma * R * TInf)
    uInf = MInf * aInf
    C0 = 0.02 * uInf * Rc

    xc, yc = (blk.xcc.get()[..., n] for n in range(2))

    r = np.sqrt(((xc - x0) ** 2 + (yc - y0) ** 2) / Rc**2)

    # the primitive vector, p, u, v, w, T
    q = np.zeros(blk.Q.shape[:3] + (mb.sim.ne,))
    # u
    q[:, :, :, 1] = uInf - (C0 * (yc - y0) / Rc**2) * np.exp(-(r**2) / 2.0)
    # v
    q[:, :, :, 2] = (C0 * (xc - x0) / Rc**2) * np.exp(-(r**2) / 2.0)

    # p
    q[:, :, :, 0] = pInf - rhoInf * C0**2 / (2.0 * Rc**2) * np.exp(-(r**2) / 2.0)
    # T
    q[:, :, :, 4] = q[:, :, :, 0] / (R * rhoInf)

    mb.set_prims([q])

    refX = xc[ng:-ng, int(NN / 2.0), ng] / Rc
    refV = np.copy(q[ng:-ng, int(NN / 2.0), ng, 2] / uInf)

    dt = 0.1 * (Lx / NE) / aInf
    tend = Lx / uInf

    ts = perf_counter()
    bar = pg.misc.Progress(tend)
    while mb.tme.get() < tend:
        if mb.nrt % 50 == 0:
            bar.at(mb.tme.get())
        mb.step(dt)
    print(f"Time integration took {perf_counter()-ts} seconds.")

    v = mb.sim.iovars.data(blk, ["v"])["v"]
    # plot v/Uinf
    plt.plot(refX, v[ng:-ng, int(NN / 2.0), ng] / uInf, label=f"{NE = }")
    plt.plot(refX, refV, "o", label="exact")
    plt.ylim([-0.016, 0.016])
    plt.xlim([-6, 6])
    plt.title("2D Euler Vortex Results")
    plt.legend()
    plt.show()
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
