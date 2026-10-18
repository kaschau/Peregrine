import numpy as np

import peregrine as pg
from peregrine.tools.base import BaseTool


def gridstats(mb):
    size = np.zeros(len(mb.blocks), dtype=np.int64)
    for blk in mb.blocks:
        size[blk.nblki] = blk.ncells
    assert np.min(size) > 0

    imax, imin = int(np.argmax(size)), int(np.argmin(size))
    big, small = mb.get_blk(imax), mb.get_blk(imin)
    return {
        "ncells": int(np.sum(size)),
        "maxblk": imax,
        "maxcells": int(np.max(size)),
        "maxext": [big.ni, big.nj, big.nk],
        "minblk": imin,
        "mincells": int(np.min(size)),
        "minext": [small.ni, small.nj, small.nk],
        "mean": float(np.mean(size)),
        "stdv": float(np.std(size)),
    }


class AnalyzeTool(BaseTool):
    name = "analyze"
    help = "report a grid's block sizes and the partitions it carries"

    def add_cli(self, parser):
        parser.add_argument("grid", help="the grid file")

    def main(self, args):
        # the extents say it all, so no coordinate is read
        mb = pg.multiblock.BaseMultiBlock.from_grid(args.grid, quiet=False)
        results = gridstats(mb)

        ni, nj, nk = results["maxext"]
        print(f"Total cells: {results['ncells']}")
        print(
            f"max block is {results['maxblk']} with {results['maxcells']}"
            f" cells, ni={ni}, nj={nj}, nk={nk}."
        )
        ni, nj, nk = results["minext"]
        print(
            f"min block is {results['minblk']} with {results['mincells']}"
            f" cells, ni={ni}, nj={nj}, nk={nk}."
        )
        print(f"mean = {results['mean']}, stdv = {results['stdv']}")

        partitions = pg.readers.GridReader(args.grid).partitions
        if partitions:
            layouts = ", ".join(f"{n}x{rpn}" for n, rpn in partitions)
            print(f"partitioned for {layouts} (ranks x ranks_per_node).")
        else:
            print("no partitions stored; balance it before a multi-rank run.")
