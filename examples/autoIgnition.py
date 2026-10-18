#!/usr/bin/env python
"""

Auto ignition flame calculation.

    python examples/autoIgnition.py <backend>

"""

from mpi4py import MPI  # noqa: F401
import sys
import peregrine as pg
import cantera as ct
import matplotlib.pyplot as plt
from pathlib import Path


def simulate(name):
    relpath = str(Path(__file__).parent)
    ct.add_directory(relpath + "/../peregrine/simulator/mixture/database/mechanisms")

    # Cantera stuff
    T, p = 1100.0, 101325
    gas = ct.Solution("CH4_O2_FFCMY.yaml")
    # set the gas state
    gas.TP = T, p
    phi = 1.0
    gas.set_equivalence_ratio(phi, "CH4", "O2")
    r1 = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r1])

    # PEREGRINE stuff
    config = pg.Config()
    config["simulation"]["simulator"] = "euler"
    config["time-integration"]["integrator"] = "rk4"
    config["mixture"]["chemistry"] = "finite-rate"
    config["chem-integration"]["kind"] = "sub-stepped"
    config["mixture"]["eos"] = "tpg"
    config["mixture"]["species"] = "CH4_O2_FFCMY.yaml"
    config["mixture"]["T-range"] = (300.0, 3500.0)
    # the reactor's state, uniform over the block
    config["ics"]["p"] = gas.P
    config["ics"]["T"] = gas.T
    config["ics"]["Y"] = {
        s: y for s, y in zip(gas.species_names, gas.Y) if y > 0.0
    }
    # a closed box: every face a slip wall
    config["bcvals"]["walls"] = {"kind": "slp-adia-wall"}
    config.validate()
    backend = pg.get_backend(name, config)
    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1], blkdims=[2, 2, 2], lengths=[0.01, 0.01, 0.01], defaultbc="walls"
    )
    mb = pg.get_solver(config, mesh, backend)

    blk = mb.blocks[0]
    ng = blk.ng

    dt = 1e-9
    config["simulation"]["dt"] = dt
    niterout = 1000
    pgT = []
    pgO2 = []
    ctT = []
    ctO2 = []
    t = []
    print(mb)
    print("Time   PEREGRINE  CANTERA")
    while mb.tme.get() < 0.05:
        if mb.nrt % niterout == 0:
            data = mb.sim.iovars.data(blk, ["T", "O2"])
            pgT.append(data["T"][ng, ng, ng])
            pgO2.append(data["O2"][ng, ng, ng])
            ctT.append(gas.T)
            ctO2.append(gas.Y[2])
            t.append(mb.tme.get())

            print(f"{mb.tme.get():.2e} {pgT[-1]:.2f} {gas.T:.2f}")

        mb.step(dt)
        sim.advance(mb.tme.get())

    plt.plot(t, pgT, label="PEREGRINE")
    plt.plot(t, ctT, label="CANTERA")
    plt.title("T [K]")
    plt.legend()
    plt.show()
    plt.plot(t, pgO2, label="PEREGRINE")
    plt.plot(t, ctO2, label="CANTERA")
    plt.title("O2")
    plt.legend()
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
