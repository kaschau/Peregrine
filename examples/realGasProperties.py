#!/usr/bin/env python
"""

Real gas properties.

    python examples/realGasProperties.py <backend>

"""

from mpi4py import MPI  # noqa: F401
import sys
import peregrine as pg
import numpy as np
import matplotlib.pyplot as plt


def simulate(name):
    config = pg.Config()
    config["simulation"]["simulator"] = "navier-stokes"
    config["mixture"]["eos"] = "real-gas"
    config["mixture"]["species"] = ["CO2"]
    config["mixture"]["trans"] = "chung"
    config["mixture"]["T-range"] = (300.0, 2000.0)
    config.validate()
    backend = pg.get_backend(name, config)
    # one block, periodic every way, so no boundary needs naming
    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1],
        blkdims=[2, 2, 2],
        lengths=[0.01, 0.01, 0.01],
        periodic=[True, True, True],
    )
    mb = pg.get_solver(config, mesh, backend)

    blk = mb.blocks[0]
    ng = blk.ng

    ps = np.linspace(25, 35, 3)
    Ts = np.linspace(600, 1600, 100)

    rhos = np.zeros((len(ps), len(Ts)))
    cps = np.zeros((len(ps), len(Ts)))
    hs = np.zeros((len(ps), len(Ts)))
    cs = np.zeros((len(ps), len(Ts)))

    mus = np.zeros((len(ps), len(Ts)))
    kappas = np.zeros((len(ps), len(Ts)))

    # the primitive vector, p, u, v, w, T, at rest
    q = np.zeros(blk.Q.shape[:3] + (mb.sim.ne,))
    for j, p in enumerate(ps):
        for i, T in enumerate(Ts):
            # the state, and everything derived from it, at this p and T
            q[:, :, :, 0] = p * 1e6
            q[:, :, :, 4] = T
            mb.set_prims([q])

            Q, state, trstate = blk.Q.get(), blk.state.get(), blk.trstate.get()
            # the eos's state at the cell by field; h = e + p / rho
            s = dict(zip(mb.sim.statefields, state[ng, ng, ng]))
            rhos[j, i] = Q[ng, ng, ng, 0]
            cps[j, i] = s["cp"]
            hs[j, i] = s["e"] + s["p"] / s["rho"]
            cs[j, i] = s["c"]

            # the transport state: mu, kappa, then a diffusivity per species
            mus[j, i] = trstate[ng, ng, ng, 0]
            kappas[j, i] = trstate[ng, ng, ng, 1]

    fig, axs = plt.subplots(2, 2, sharex=True)
    fig.suptitle(f"Thermo Properties of {config['mixture']['species'][0]}")
    axs[0, 0].set_ylabel("rho [kg/m^3]")
    axs[0, 1].set_ylabel("Cp [J/kg.K]")
    axs[1, 0].set_ylabel("h [J/kg]")
    axs[1, 1].set_ylabel("c [m/s]")
    for i, p in enumerate(ps):
        axs[0, 0].plot(Ts, rhos[i, :], label=f"p={p} MPa")
        axs[0, 1].plot(Ts, cps[i, :], label=f"p={p} MPa")
        axs[1, 0].plot(Ts, hs[i, :], label=f"p={p} MPa")
        axs[1, 1].plot(Ts, cs[i, :], label=f"p={p} MPa")
    axs[1, 0].set_xlabel("Temperature [K]")
    axs[1, 1].set_xlabel("Temperature [K]")
    for ax in axs.ravel():
        ax.legend()
        ax.grid()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle(f"Transport Properties of {config['mixture']['species'][0]}")
    ax2.set_xlabel("Temperature [K]")
    ax1.set_ylabel("Thermal Cond. [W/m.K]")
    ax2.set_ylabel("Viscosity [Pa.s]")
    for i, p in enumerate(ps):
        ax1.plot(Ts, kappas[i, :], label=f"p={p} MPa")
        ax2.plot(Ts, mus[i, :], label=f"p={p} MPa")
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    plt.show()
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
