#!/usr/bin/env python
"""

Generate 1D detonation profile

    python examples/detonationProfile.py <backend>

"""

from mpi4py import MPI  # noqa: F401
import sys
from pathlib import Path

import cantera as ct

import matplotlib.pyplot as plt
import numpy as np
import peregrine as pg


def simulate(name):
    relpath = str(Path(__file__).parent)
    ct.add_directory(relpath + "/../peregrine/simulator/mixture/database/mechanisms")
    gas = ct.Solution("CH4_O2_FFCMY.yaml")
    # set the gas state
    gas.TP = 300.0, 101325.0
    phi = 1.0
    gas.set_equivalence_ratio(phi, "CH4", "O2")

    config = pg.Config()
    config["simulation"]["simulator"] = "euler"
    # kepaec blended with rusanov where Jameson's pressure sensor fires
    config["adv-flux"]["formulation"] = "blended"
    config["adv-flux"]["primary"] = "kepaec"
    config["adv-flux"]["secondary"] = "rusanov"
    config["adv-flux"]["switch"] = "jameson"
    config["adv-flux"]["values"] = {"p-gain": 5.0, "gamma-gain": 0.0}
    config["time-integration"]["integrator"] = "tvd-rk3"
    config["mixture"]["chemistry"] = "finite-rate"
    config["chem-integration"]["kind"] = "sub-stepped"
    config["mixture"]["eos"] = "tpg"
    config["mixture"]["species"] = "CH4_O2_FFCMY.yaml"
    config["mixture"]["T-range"] = (300.0, 3500.0)
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    config.validate()
    backend = pg.get_backend(name, config)

    nx = 300
    dx = 0.005 / 50.0  # Aproximate rde resolution
    lx = nx * dx
    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1],
        blkdims=[nx, 2, 2],
        lengths=[lx, 0.01, 0.01],
        bcnames=dict.fromkeys(range(1, 7), "walls"),
    )
    mb = pg.get_solver(config, mesh, backend)

    blk = mb.blocks[0]
    ng = blk.ng

    # the primitive vector, p, u, v, w, T, Y: the mixture at rest, and hot
    # and pressed behind the shock
    q = np.zeros(blk.Q.shape[:3] + (mb.sim.ne,))
    q[:, :, :, 0] = gas.P
    q[:, :, :, 4] = gas.T
    q[:, :, :, 5::] = gas.Y[0:-1]
    xc = blk.xcc.get()[..., 0]
    shockX = lx * 0.05
    q[:, :, :, 0] = np.where(xc < shockX, 4.0e6, q[:, :, :, 0])
    q[:, :, :, 4] = np.where(xc < shockX, 3000.0, q[:, :, :, 4])
    mb.set_prims([q])

    dt = 1.0e-9
    config["simulation"]["dt"] = dt
    testIndex = int(nx / 2)
    print(mb)
    bar = pg.misc.Progress(testIndex)
    T = lambda: mb.sim.iovars.data(blk, ["T"])["T"][:, ng, ng]
    while T()[testIndex] < 350.0:
        if mb.nrt % 10 == 0:
            detLoc = np.where(T() > 350.0)[0][-1]
            bar.at(detLoc)
        if not np.isfinite(blk.Q.get()[blk.interior]).all():
            print("Nan")
            break

        mb.step(dt)

    data = mb.sim.iovars.data(blk, ["p", "u", "T", "O2", "H2O", "CO2", "CH4"])
    line = lambda name: data[name][ng:-ng, ng, ng]
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_title("1D Detonation Profile")
    ax1.set_ylabel("Pressure [MPa]")
    ax1.set_xlabel(r"x")
    x = blk.xcc.get()[..., 0][ng:-ng, ng, ng]
    p = line("p") / 1e6
    ax1.plot(x, p, color="r", label="p", linewidth=0.5)
    ax12 = ax1.twinx()
    ax12.set_ylabel("Temperatur[K] / Velocity [m/s]")
    u = line("u")
    T = line("T")
    ax12.plot(x, T, color="k", label="T", linewidth=0.5)
    ax12.plot(x, u, color="g", label="u", linewidth=0.5)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax12.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2)

    O2, H2O, CO2, CH4 = (line(name) for name in ("O2", "H2O", "CO2", "CH4"))
    ax2.plot(x, O2, color="b", label="O2", linewidth=0.5)
    ax2.plot(x, CH4, color="r", label="CH4", linewidth=0.5)
    ax2.plot(x, H2O, color="k", label="H2O", linewidth=0.5)
    ax2.plot(x, CO2, color="g", label="CO2", linewidth=0.5)
    ax2.legend()
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
