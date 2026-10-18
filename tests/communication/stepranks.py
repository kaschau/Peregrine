import sys

import numpy as np
from mpi4py import MPI

import peregrine as pg

# a script, not a test module: mpiexec -n <ranks> python stepranks.py <grid> <ranks> <out> <simulator> <recon> <backend>
grid, ranks, out, simulator, recon, name = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
config = pg.Config()
config["simulation"]["simulator"] = simulator
config["mixture"]["species"] = {
    "N2": {"MW": 28.0134, "cp0": 1040.0, "mu0": 1.8e-5, "kappa0": 0.026},
    "O2": {"MW": 31.9988, "cp0": 918.0, "mu0": 2.0e-5, "kappa0": 0.027},
}
config["mixture"]["eos"] = "cpg"
if simulator == "navier-stokes":
    config["mixture"]["trans"] = "constant"
config["ics"]["Y"] = {"N2": 0.77}
config["bcvals"]["wall"] = {"kind": "slp-adia-wall"}
config["time-integration"]["integrator"] = "tvd-rk3"
if recon == "muscl":
    config["adv-flux"].update(recon="muscl", limiter="koren")
backend = pg.get_backend(name, config)
dt = {"euler": 2e-6, "navier-stokes": 1e-4}[simulator]
comm = MPI.COMM_WORLD


def assert_landing(mb):
    ng = mb.ng
    for blk in mb.blocks:
        Q = blk.Q.get()
        i, j, k = np.indices(Q.shape[:3])
        Q[...] = (1000 * blk.nblki + 100 * i + 10 * j + k)[..., None]
        blk.Q.set(Q)
    g = mb.backend.graph("stamp")
    g.add_send("sent", "Q")
    g.add_recv("landed", "Q")
    g.bind(mb)
    g.run()
    for blk in mb.blocks:
        Q = blk.Q.get()
        j, k = np.indices(Q.shape[1:3])
        for face in blk.intfaces:
            if not face.offrank:
                continue
            # the exchange fills the interface proper only, so the halo's edges keep the block's own stamp
            halo = face.halo(Q)[:, ng:-ng, ng:-ng, 0]
            if face.is_low:
                # block 0's last interior planes, layer 0 the last of them
                expected = [100 * (Q.shape[0] - ng - 1 - g) + 10 * j + k for g in range(ng)]
            else:
                expected = [1000 + 100 * (ng + g) + 10 * j + k for g in range(ng)]
            assert np.array_equal(halo, np.stack(expected)[:, ng:-ng, ng:-ng]), (blk.nblki, face.nface)


try:
    mb = pg.get_solver(config, pg.readers.GridReader(grid, ranks=(ranks, ranks)), backend)
    if ranks > 1 and not any(f.orient != "123" for _, f in mb.intfaces() if f.nbr is not None):
        assert_landing(mb)
    prims = []
    for blk in mb.blocks:
        x, y, z = (blk.xcc.get()[..., c] for c in range(3))
        bump = np.exp(-(((x - 1.0) ** 2 + (y - 0.4) ** 2 + (z - 0.6) ** 2) / 0.1))
        q = np.zeros(x.shape + (mb.sim.ne,))
        q[..., 0] = 101325.0 * (1.0 + 0.05 * bump)
        q[..., 1] = 30.0 * np.sin(2 * np.pi * y) * bump
        q[..., 2] = -20.0 * np.cos(2 * np.pi * z) * bump
        q[..., 3] = 10.0 * bump
        q[..., 4] = 300.0 * (1.0 + 0.1 * bump)
        q[..., 5] = 0.77 - 0.2 * bump
        prims.append(q)
    mb.set_prims(prims)
    for _ in range(5):
        mb.step(dt)
    mine = {str(blk.nblki): blk.Q.get()[blk.interior].copy() for blk in mb.blocks}
    gathered = comm.gather(mine, root=0)
    if comm.rank == 0:
        blocks = {}
        for part in gathered:
            blocks.update(part)
        np.savez(out, **blocks)
finally:
    backend.stop()
