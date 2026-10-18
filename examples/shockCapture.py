#!/usr/bin/env python3

"""
Solves the shock tube problem defined with arbutrary left and right states states separated by a membrane
at some x location between zero and one.

Solves the problem numerically with peregrine, and exactly using an exact Riemann solver.

See

Riemann Solvers and Numerical Methods for Fluid Dynamic 3rd Ed.
Eleuterio F. Toro
Spring

for more.

    python examples/shockCapture.py <backend>
"""

from mpi4py import MPI  # noqa: F401

import sys
from pathlib import Path

import peregrine as pg
import numpy as np
import matplotlib.pyplot as plt

# the exact solution lives with the tests that measure against it
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.riemann import RiemannProblem  # noqa: E402

# the debug gas of the Toro cases: R = 281.4, gamma = 1.4
db = {"DB": {"MW": 29.54065178914549, "cp0": 1000.0}}


def simulate(name, testnum, index="i"):
    nx = 201
    config = pg.Config()
    config["simulation"]["simulator"] = "euler"
    config["mixture"]["species"] = db
    config["adv-flux"]["formulation"] = "blended"
    config["adv-flux"]["primary"] = "kepaec"
    config["adv-flux"]["secondary"] = "rusanov"
    # Jameson's pressure sensor alone: the gamma sensor off
    config["adv-flux"]["switch"] = "jameson"
    config["adv-flux"]["values"] = {"p-gain": 5.0, "gamma-gain": 0.0}
    config["time-integration"]["integrator"] = "tvd-rk3"

    rot = {"i": 0, "j": 1, "k": 2}

    def rotate(li, index):
        return li[-rot[index] :] + li[: -rot[index]]

    blkdims = rotate([nx, 2, 2], index)
    lengths = rotate([1, 0.1, 0.1], index)

    # the gas, from its constants
    species = db[next(iter(db))]
    R = 8314.46261815324 / species["MW"]
    cp = species["cp0"]
    gamma = cp / (cp - R)

    test = RiemannProblem.toro(testnum, gamma, R)
    print("State {}".format(testnum))
    print("--------------------------")
    print("Left State")
    print("PL = {}".format(test.pL))
    print("TL = {}".format(test.TL))
    print("rhoL = {}".format(test.rhoL))
    print("uL = {}".format(test.uL))
    print("--------------------------")
    print("Right State")
    print("PR = {}".format(test.pR))
    print("TR = {}".format(test.TR))
    print("rhoR = {}".format(test.rhoR))
    print("uR = {}".format(test.uR))
    print("--------------------------")

    # the ends of the tube: fed where the flow comes in, let out where it
    # leaves, walls where it is still; the sides slip
    ccAxis = {"i": 0, "j": 1, "k": 2}
    lo_intface, hi_intface = 2 * ccAxis[index] + 1, 2 * ccAxis[index] + 2
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    bcnames = dict.fromkeys(range(1, 7), "walls")

    def end(name, u, p, T, nface):
        if u == 0.0:
            return
        bcnames[nface] = name
        # in through the low end, or in through the high end, is an inlet
        if (u > 0) == (nface == lo_intface):
            velo = rotate([u, 0.0, 0.0], index)
            config["bcvals"][name] = {
                "kind": "sub-in-fvt",
                "u": velo[0],
                "v": velo[1],
                "w": velo[2],
                "T": T,
            }
        else:
            config["bcvals"][name] = {
                "kind": "sub-out-fp",
                "p": p,
            }

    end("left", test.uL, test.pL, test.TL, lo_intface)
    end("right", test.uR, test.pR, test.TR, hi_intface)
    config.validate()
    backend = pg.get_backend(name, config)

    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1],
        blkdims=blkdims,
        lengths=lengths,
        bcnames=bcnames,
    )
    mb = pg.get_solver(config, mesh, backend)
    print(mb)

    blk = mb.blocks[0]
    ng = blk.ng

    uIndex = {"i": 1, "j": 2, "k": 3}
    xc = blk.xcc.get()[..., ccAxis[index]]
    # the primitive vector, p, u, v, w, T: the left and right states
    q = np.zeros(blk.Q.shape[:3] + (mb.sim.ne,))
    q[:, :, :, 0] = np.where(xc <= test.x0, test.pL, test.pR)
    q[:, :, :, uIndex[index]] = np.where(xc <= test.x0, test.uL, test.uR)
    q[:, :, :, 4] = np.where(xc <= test.x0, test.TL, test.TR)
    mb.set_prims([q])

    bar = pg.misc.Progress(test.t)
    while mb.tme.get() < test.t:
        bar.at(mb.tme.get())
        mb.step(test.dt)

    s_ = rotate(np.s_[ng:-ng, ng, ng], index)
    data = mb.sim.iovars.data(blk, ["rho", "p", "u", "v", "w", "T"])
    x = blk.xcc.get()[..., ccAxis[index]][s_]
    rho = data["rho"][s_]
    p = data["p"][s_]
    u = data["uvw"[ccAxis[index]]][s_]
    # the internal energy on the exact solution's reference, zero at 0 K; the eos's is referenced at 298.15 K
    e = (cp - R) * data["T"][s_]

    rx = np.linspace(0, 1, 250)
    res = test.solve(rx)
    rrho = res["rho"]
    ru = res["u"]
    rp = res["p"]
    re = res["energy"]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    ax1 = ax[0][0]
    ax2 = ax[0][1]
    ax3 = ax[1][0]
    ax4 = ax[1][1]

    ax1.set_title(f"{mb.tme.get():.2f}")

    ms = 1.5
    lw = 0.5
    # rho
    ax1.set_xlabel(r"x")
    ax1.plot(x, rho, color="k", label="rho", linewidth=lw)
    ax1.scatter(rx, rrho, color="k", label="Analyticsl", marker="o", s=ms)
    ax1.legend()

    # velocity
    ax2.set_xlabel(r"x")
    ax2.plot(x, u, color="k", label="u", linewidth=lw)
    ax2.scatter(rx, ru, color="k", label="Analytical", marker="o", s=ms)
    ax2.legend()

    # pressure
    ax3.set_xlabel(r"x")
    ax3.plot(x, p, color="k", label="p", linewidth=lw)
    ax3.scatter(rx, rp, color="k", label="Analytical", marker="o", s=ms)
    ax3.legend()

    # energy
    ax4.set_xlabel(r"x")
    ax4.plot(x, e, color="k", label="e", linewidth=lw)
    ax4.scatter(rx, re, color="k", label="Analyticsl", marker="o", s=ms)
    ax4.legend()

    plt.show()
    plt.close()
    backend.stop()


if __name__ == "__main__":
    try:
        testnum = 5
        index = "i"
        simulate(sys.argv[1], testnum, index)

    except Exception as e:
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
