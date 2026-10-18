import os
from pathlib import Path

from peregrine.misc import get_comm_rank_size
from peregrine.writers import GridWriter, RestartWriter
from peregrine.plugins.solver.base import BaseSolverPlugin


class Writer(BaseSolverPlugin):
    name = "writer"

    def start(self, solver):
        cfgsect = self.cfgsect
        comm, rank, size = get_comm_rank_size()
        self.dir = Path(cfgsect.get("dir", "."))
        if rank == 0:
            self.dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

        precision = cfgsect.get("precision", "single")
        gridf = solver.mesh.fname
        if gridf is None:
            gridf = self.dir / "g.h5"
            GridWriter(solver, str(gridf), precision).write(solver)
        basename = cfgsect.get("basename", "q.{n:08d}")
        self.restart = RestartWriter(
            solver,
            solver.sim.iovars,
            str(self.dir / f"{basename}.h5"),
            os.path.relpath(gridf, self.dir),
            precision,
            cfg=solver.cfg,
        )

    def __call__(self, solver):
        self.restart.write(solver, solver.nrt, solver.tme.get())
