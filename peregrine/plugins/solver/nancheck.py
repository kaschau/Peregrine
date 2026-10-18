import numpy as np
from mpi4py import MPI

from peregrine.launch.kernel import Kernel
from peregrine.misc import get_comm_rank_size
from peregrine.plugins.solver.base import BaseSolverPlugin
from peregrine.plugins.solver.writer import Writer


class NanCheck(BaseSolverPlugin):
    name = "nancheck"

    def start(self, solver):
        soln = solver.regs[0]
        options = dict(ncomps=solver.columns[soln].comps[0])
        tpl = f"{__package__}.kernels.allfinite"
        self.kernel = Kernel(solver.provider, tpl, options=options, Q=soln)

    def __call__(self, solver):
        comm, rank, size = get_comm_rank_size()
        soln = solver.regs[0]
        finite = solver.launch(self.kernel, "interior")
        bad = np.array([finite[0] < 0.5], np.int32)
        comm.Allreduce(MPI.IN_PLACE, bad, op=MPI.SUM)
        if not bad[0]:
            return
        for blk in solver.blocks:
            Q = getattr(blk, soln).get()[blk.interior]
            where = np.where((~np.isfinite(Q)).any(axis=-1))
            if len(where[0]) == 0:
                continue
            cells = blk.xcc.get()[blk.interior]
            with open(f"nans_{blk.nblki}.log", "w") as f:
                f.write(f"Non-finite state, block {blk.nblki}\n")
                for x, y, z in cells[where]:
                    f.write(f"x = {x} y = {y} z = {z}\n")
        # the state it stopped on, written by the writer among the plugins, if there is one
        for plugin in solver.plugins.values():
            if isinstance(plugin, Writer):
                plugin(solver)
        comm.Barrier()
        raise RuntimeError(
            f"non-finite state at step {solver.nrt}; see the nans_<block>.log files"
        )
