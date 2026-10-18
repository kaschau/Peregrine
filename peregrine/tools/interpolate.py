import os
from pathlib import Path

import peregrine as pg
from peregrine.tools.base import BaseTool


class InterpolateTool(BaseTool):
    name = "interpolate"
    help = "interpolate a result onto another grid"

    def add_cli(self, parser):
        parser.add_argument(
            "result",
            help="the result to interpolate from, which names its own grid",
        )
        parser.add_argument("grid", help="the grid file to interpolate onto")
        parser.add_argument("out", help="the result to write")
        parser.add_argument(
            "-cfg", help="the case's yaml, for a result that carries none"
        )
        parser.add_argument(
            "-kind",
            default="nearest",
            help="nearest, or rbf: a radial basis function through the samples",
        )
        parser.add_argument(
            "-basis",
            help="an rbf's basis, one scipy knows: linear, cubic, ... (linear)",
        )
        parser.add_argument(
            "-smooth",
            type=float,
            help="an rbf's smoothing; larger is smoother (0.5)",
        )
        parser.add_argument(
            "-deep_search",
            action="store_true",
            help="search every block for the ones a block lies in: slower,"
            " and surer where blocks are curved",
        )

    def main(self, args):
        reader = pg.readers.RestartReader(args.result, quiet=False)
        cfg = pg.read_cfg(args.cfg) if args.cfg else reader.cfg
        if cfg is None:
            raise SystemExit(
                "peregrine interpolate: the result carries no config;"
                " give one with -cfg"
            )
        # the vector alone: a result carries nothing a tool could derive
        iovars = pg.simulator.get_simulator(cfg).iovars.minimal
        nprims = len(iovars.privars)
        src = pg.multiblock.Restart(nprims)
        pg.readers.GridReader(reader.grid, quiet=False).fill(src)
        reader.fill(src, iovars)
        tgt = pg.multiblock.Restart.from_grid(args.grid, nprims, quiet=False)

        # the kind's own options, those given; a kind that does not take one refuses it
        given = {"basis": args.basis, "smooth": args.smooth}
        options = {key: value for key, value in given.items() if value is not None}
        interp = pg.interpolation.get_interpolator(args.kind, deep_search=args.deep_search, **options)
        interp.interpolate(src, tgt)

        # the result names its grid relative to itself
        out = Path(args.out)
        gridf = os.path.relpath(args.grid, out.parent)
        writer = pg.writers.RestartWriter(tgt, iovars, out, gridf, quiet=False)
        writer.write(tgt, reader.nrt, reader.tme)
