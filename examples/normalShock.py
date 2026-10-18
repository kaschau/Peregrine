#!/usr/bin/env python
"""

Generate 1D normal shock

    python examples/normalShock.py <backend>

"""

from mpi4py import MPI  # noqa: F401

import sys

import matplotlib.pyplot as plt
import numpy as np
import peregrine as pg

# a calorically perfect air, stated in full: the library carries no such species
air = {
    "Air": {
        "MW": 28.97,
        "cp0": 1002.838449439523,
        "mu0": 1.8591191080521142e-05,
        "kappa0": 0.02625394405190068,
    }
}


##################################################
######### 1D Normal Shock ########################
##################################################

# Set upstream values here
M1 = 2.0
p1 = 101325.0
T1 = 300.0


def simulate(name):
    config = pg.Config()
    config["simulation"]["simulator"] = "euler"
    config["mixture"]["eos"] = "cpg"
    config["mixture"]["species"] = air
    config["adv-flux"]["primary"] = "rusanov"
    config["time-integration"]["integrator"] = "rk4"

    # the gas, and the post shock state in the lab frame, from the
    # calorically perfect relations
    cp = air["Air"]["cp0"]
    R = 8314.46261815324 / air["Air"]["MW"]
    gamma = cp / (cp - R)
    c1 = np.sqrt(gamma * R * T1)
    rho1 = p1 / (R * T1)
    M2 = np.sqrt((M1**2 * (gamma - 1) + 2) / (2 * gamma * M1**2 - (gamma - 1)))
    T2 = T1 * (
        ((1 + (gamma - 1) / 2 * M1**2) * (2 * gamma / (gamma - 1) * M1**2 - 1))
        / (M1**2 * (2 * gamma / (gamma - 1) + (gamma - 1) / 2))
    )
    p2 = p1 * (2 * gamma * M1**2 / (gamma + 1) - (gamma - 1) / (gamma + 1))
    c2 = np.sqrt(gamma * R * T2)
    u2 = -M2 * c2 + M1 * c1  # In lab reference frame

    # the inlet feeds the post shock state; every other side slips
    config["bcvals"]["inlet"] = {
        "kind": "sub-in-fvt",
        "u": u2,
        "v": 0.0,
        "w": 0.0,
        "T": T2,
    }
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    config.validate()
    backend = pg.get_backend(name, config)

    nx = 300
    lx = 1.0
    dx = lx / nx
    bcnames = dict.fromkeys(range(1, 7), "walls")
    bcnames[1] = "inlet"
    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1],
        blkdims=[nx, 2, 2],
        lengths=[lx, 0.01, 0.01],
        bcnames=bcnames,
    )
    mb = pg.get_solver(config, mesh, backend)
    blk = mb.blocks[0]
    ng = blk.ng

    # the primitive vector, p, u, v, w, T: upstream everywhere, the post
    # shock state behind the shock
    q = np.zeros(blk.Q.shape[:3] + (mb.sim.ne,))
    q[:, :, :, 0] = p1
    q[:, :, :, 4] = T1
    xc = blk.xcc.get()[..., 0]
    shockX = lx * 0.05
    q[:, :, :, 0] = np.where(xc < shockX, p2, q[:, :, :, 0])
    q[:, :, :, 1] = np.where(xc < shockX, u2, q[:, :, :, 1])
    q[:, :, :, 4] = np.where(xc < shockX, T2, q[:, :, :, 4])
    mb.set_prims([q])

    # Set dt based on cfg estimate
    dt = 0.25 * dx / (c2 + u2)
    testIndex = int(nx / 2)
    print(mb)
    bar = pg.misc.Progress(testIndex)
    T = lambda: mb.sim.iovars.data(blk, ["T"])["T"][:, ng, ng]
    while T()[testIndex] < 301.0:
        if mb.nrt % 10 == 0:
            shockLoc = np.where(T() > 301.0)[0][-1]
            bar.at(shockLoc)
        if not np.isfinite(blk.Q.get()[blk.interior]).all():
            print("Nan")
            break

        mb.step(dt)

    data = mb.sim.iovars.data(blk, ["rho", "p", "u", "T"])
    fig, ax1 = plt.subplots()
    ax1.set_title("1D Normal Shock")
    ax1.set_ylabel("p/p1")
    ax1.set_xlabel(r"x")
    x = blk.xcc.get()[..., 0][ng:-ng, ng, ng]
    p = data["p"][ng:-ng, ng, ng] / p1
    ax1.plot(x, p, color="r", label="p2/p1", linewidth=0.5)
    ax2 = ax1.twinx()
    ax2.set_ylabel("T, M , rho")
    # convert back to shock reference frame
    c = np.sqrt(gamma * R * data["T"][ng:-ng, ng, ng])
    u = -(data["u"][ng:-ng, ng, ng] - M1 * c1) / c
    T = data["T"][ng:-ng, ng, ng] / T1
    rho = data["rho"][ng:-ng, ng, ng] / rho1
    ax2.plot(x, T, color="k", label="T/T1", linewidth=0.5)
    ax2.plot(x, u, color="g", label="M", linewidth=0.5)
    ax2.plot(x, rho, color="orange", label="rho/rho1", linewidth=0.5)
    ax2.set_ylim([0, None])

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2)
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
