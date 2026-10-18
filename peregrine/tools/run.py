import traceback

import numpy as np

import peregrine as pg
from peregrine.tools.base import BaseTool


def simulate(args):
    comm, rank, size = pg.misc.get_comm_rank_size()
    # a result carries the case and the grid it came from; either given here wins
    state = pg.readers.RestartReader(args.restart) if args.restart else None
    cfg = pg.read_cfg(args.cfg) if args.cfg else state.cfg
    backend = pg.get_backend(args.backend, cfg)
    ranks = (size, pg.misc.get_ranks_per_node())
    mesh = pg.readers.GridReader(args.grid or state.grid, ranks)
    pg.get_solver(cfg, mesh, backend, state).run()
    backend.stop()


class RunTool(BaseTool):
    name = "run"
    help = (
        "run a case from a config and a grid,"
        " or from a result, which carries both"
    )

    def add_cli(self, parser):
        parser.add_argument("cfg", nargs="?", help="the case's yaml")
        parser.add_argument("grid", nargs="?", help="the grid file")
        parser.add_argument("-r", "--restart", help="a result to restart from")
        names = sorted(c.name for c in pg.misc.subclasses(pg.backend.BaseBackend) if c.name)
        parser.add_argument("-b", "--backend", required=True, choices=names, help="where the case runs")

    def main(self, args):
        from mpi4py import MPI

        if not args.restart and not (args.cfg and args.grid):
            raise SystemExit(
                "peregrine run: a config and a grid,"
                " or a result to restart from"
            )
        # underflow is harmless and a trace mass fraction over rho does it; a raise on some ranks leaves the rest waiting in a collective write
        np.seterr(all="raise", under="ignore")
        try:
            simulate(args)
        except Exception:
            # one rank's failure ends the run, not just that rank
            traceback.print_exc()
            MPI.COMM_WORLD.Abort(1)
