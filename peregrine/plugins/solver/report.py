from time import perf_counter

from mpi4py import MPI

from peregrine.misc import get_comm_rank_size
from peregrine.plugins.solver.base import BaseSolverPlugin


class Report(BaseSolverPlugin):
    name = "report"

    banner = (
        " >>> ******************************** <<<\n"
        "              PEREGRINE CFD\n"
        " >>> ******************************** <<<\n"
        "  Copyright (c) 2021 Kyle A. Schau\n"
        "           All rights reserved.\n"
    )

    def start(self, solver):
        # a restart's nrt counts the whole case; this run's time is spread over its own steps
        self.t0, self.nrt0 = perf_counter(), solver.nrt
        if get_comm_rank_size()[1] == 0:
            print(self.banner)
            print(solver)

    def __call__(self, solver):
        # cflmax reduces over the ranks, so every rank calls it before the rank check
        cfl = solver.cflmax()
        if get_comm_rank_size()[1] != 0:
            return
        dt = solver.dt.get()
        names = [n for n in solver.sim.cflspeeds if n != "combined"]
        rows = "".join(f"         {n:10s}: {cfl[n] * dt:.3f}\n" for n in names)
        print(
            f" >>> --------- nrt: {solver.nrt:<6} ---------- <<<\n"
            f"     tme: {solver.tme.get():.6E} s\n"
            f"     dt : {dt:.6E} s\n"
            f"     MAX CFL       : {cfl['combined'] * dt:.3f}\n"
            + rows
            + " >>> -------------------------------- <<<\n",
        )

    def nsteps(self, solver):
        return max(solver.nrt - self.nrt0, 1)

    def finalize(self, solver):
        comm, rank, size = get_comm_rank_size()
        elapsed = perf_counter() - self.t0
        cells = comm.allreduce(solver.ncells, op=MPI.SUM)
        if rank != 0:
            return
        hrs, rem = divmod(int(elapsed), 3600)
        mins, secs = divmod(rem, 60)
        steps = self.nsteps(solver)
        print(
            "PEREGRINE simulation completed.\n"
            f"Simulation time: {hrs}h : {mins}m : {secs}s over {steps} steps\n"
            f"Seconds/Iteration/Cell: {elapsed / steps / cells:.3e}\n"
        )
