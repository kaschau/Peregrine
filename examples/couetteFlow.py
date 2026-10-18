#!/usr/bin/env python
"""

Couette Flow with top wall moving at 5m/s

    python examples/couetteFlow.py <backend>

"""

from mpi4py import MPI  # noqa: F401

import sys

import peregrine as pg
import numpy as np
import matplotlib.pyplot as plt

# a calorically perfect air, stated in full: the library carries no such species
air = {
    "Air": {
        "MW": 28.97,
        "cp0": 1002.838449439523,
        "mu0": 1.8591191080521142e-05,
        "kappa0": 0.02625394405190068,
    }
}


def analytical(y, h, t, nu, wallSpeed):
    n = np.array([i for i in range(50)][1::])
    mult = wallSpeed / abs(wallSpeed)
    return mult * (
        wallSpeed * y / h
        - 2
        * wallSpeed
        / np.pi
        * np.sum(
            1.0
            / n
            * np.exp(-(n**2) * np.pi**2 * nu * t / h**2)
            * np.sin(n * np.pi * (1 - y / h))
        )
    )


def simulate(name, index, velo):
    wallSpeed = 5.0
    nx = 50
    h = 0.025
    if "-" in velo:
        wallSpeed *= -1.0

    if index == "i":
        assert ("y" in velo) or ("z" in velo)
    elif index == "j":
        assert ("x" in velo) or ("z" in velo)
    elif index == "k":
        assert ("y" in velo) or ("x" in velo)

    config = pg.Config()
    # the explicit step is the acoustic one, c dt/dx about 0.7, so this Mach 0.015 case takes two million of them
    config["simulation"]["dt"] = 1e-6
    config["simulation"]["simulator"] = "navier-stokes"
    config["mixture"]["trans"] = "constant"
    config["mixture"]["species"] = air

    rot = {"i": 0, "j": 1, "k": 2}

    def rotate(li, index):
        return li[-rot[index] :] + li[: -rot[index]]

    blkdims = rotate([nx, 2, 2], index)
    lengths = rotate([h, 0.001, 0.001], index)
    flowAxis = "xyz".index(velo[-1])
    flowVelocity = "uvw"[flowAxis]
    periodic = [n == flowAxis for n in range(3)]

    # the low side of the wall-normal axis is still, the high side moves,
    # and the sides across the flow slip
    wallAxis = rot[index]
    still, moving = 2 * wallAxis + 1, 2 * wallAxis + 2
    bcnames = {n: "slip" for n in range(1, 7)}
    bcnames[still], bcnames[moving] = "still", "moving"
    wallVelocity = {c: wallSpeed if c == flowVelocity else 0.0 for c in "uvw"}
    config["bcvals"]["still"] = {"kind": "no-slp-adia-wall"}
    config["bcvals"]["moving"] = {"kind": "mov-adia-wall", **wallVelocity}
    config["bcvals"]["slip"] = {"kind": "slp-adia-wall"}
    config.validate()
    backend = pg.get_backend(name, config)

    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1],
        blkdims=blkdims,
        lengths=lengths,
        periodic=periodic,
        bcnames=bcnames,
    )
    mb = pg.get_solver(config, mesh, backend)
    blk = mb.blocks[0]
    ng = blk.ng

    mu = air["Air"]["mu0"]
    rho = blk.Q.get()[ng, ng, ng, 0]
    nu = mu / rho

    if index == "i":
        s_ = np.s_[ng:-ng, ng, ng]
    elif index == "j":
        s_ = np.s_[ng, ng:-ng, ng]
    elif index == "k":
        s_ = np.s_[ng, ng, ng:-ng]
    xc = blk.xcc.get()[..., wallAxis][s_]

    outputTimes = [0.0005, 0.005, 0.05]
    doneOutput = [False for _ in range(len(outputTimes))]
    outputU = []
    simTme = max(outputTimes) * h**2 / nu
    bar = pg.misc.Progress(simTme)
    while mb.tme.get() < simTme:
        mb.step(config["simulation"]["dt"])

        if mb.nrt % 20 == 0:
            bar.at(mb.tme.get())
            if np.any(np.isnan(blk.Q.get())):
                raise ValueError("Nan detected")

        for i, oT in enumerate(outputTimes):
            t = oT * h**2 / nu
            if mb.tme.get() >= t and not doneOutput[i]:
                outputU.append(mb.sim.iovars.data(blk, [flowVelocity])[flowVelocity][s_])
                doneOutput[i] = True

    # Analytical solution
    yplot = np.linspace(0, h, 100)
    anSol = []
    for oT, oU in zip(outputTimes, outputU):
        sol = []
        for yy in yplot:
            t = oT * h**2 / nu
            sol.append(analytical(yy, h, t, nu, wallSpeed))
        anSol.append(np.array(sol))
        exact = np.array([analytical(yy, h, t, nu, wallSpeed) for yy in xc])
        err = np.abs(oU - exact).max() / abs(wallSpeed)
        print(f"t nu/h^2 = {oT}: max |u - exact| / U = {err:.3e}")

    fig, ax1 = plt.subplots()
    ax1.grid(True, linestyle="--")
    ax1.set_title("Couette Results")
    ax1.set_xlabel(r"$u/U$")
    ax1.set_ylabel(r"$y/h$")
    y = xc / h
    y = np.append(y, [1.0])
    legends = [str(i) for i in outputTimes]
    for oU, oA, legend in zip(outputU, anSol, legends):
        ax1.scatter(
            np.append(oU, [wallSpeed]) / abs(wallSpeed), y, label=legend, s=15.0
        )
        ax1.plot(oA / wallSpeed, yplot / h, linewidth=0.5, color="k")
    ax1.scatter(
        np.linspace(0, wallSpeed / abs(wallSpeed), y.shape[0]),
        np.linspace(0, 1, y.shape[0]),
        marker="o",
        facecolor="None",
        edgecolor="b",
        label="Steady State",
        linewidth=0.5,
    )
    ax1.legend(title=r"$h^2t/\nu$")
    plt.show()
    plt.close()
    backend.stop()


if __name__ == "__main__":
    try:
        index = "j"
        velo = "+z"
        simulate(sys.argv[1], index, velo)

    except Exception as e:
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
