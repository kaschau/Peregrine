import numpy as np

import peregrine as pg
from peregrine.partition import get_partitioner
from peregrine.tools.base import BaseTool


class PartitionTool(BaseTool):
    name = "partition"
    help = "add a partition for so many ranks to a grid file, load balanced"

    def add_cli(self, parser):
        parser.add_argument("grid", help="the grid file")
        parser.add_argument(
            "-ranks",
            type=int,
            required=True,
            help="how many ranks to balance for",
        )
        parser.add_argument(
            "-ranks_per_node",
            type=int,
            required=True,
            help="how many of those ranks share a node: blocks are split over "
            "nodes first, so the network carries as little as possible, then "
            "over the ranks within a node",
        )
        parser.add_argument(
            "-method",
            default="auto",
            help="which partitioner places the blocks: auto, greedy or metis",
        )
        parser.add_argument(
            "-halocost",
            type=float,
            help="what a halo cell behind any face costs a rank, in interior "
            "cells; omit it for the measured default",
        )
        parser.add_argument(
            "-tradecost",
            type=float,
            help="what a halo cell behind a traded interface costs besides, "
            "its pack, unpack and fill; omit it for the measured default",
        )
        parser.add_argument(
            "-ng",
            type=int,
            help="how deep the case's halo is: one unless said, two with a "
            "switch or a reconstruction",
        )
        parser.add_argument(
            "-granularity",
            type=float,
            help="cut blocks down before placing them: the cap on a piece is "
            "an even share of the cells divided by this, so 1 cuts only what "
            "cannot fit a rank and 2 aims for two pieces per rank; omit it to "
            "place the blocks uncut",
        )
        parser.add_argument(
            "-plot",
            action="store_true",
            help="show the load and block count on each rank when done",
        )

    def main(self, args):
        ranks, ranks_per_node = args.ranks, args.ranks_per_node
        granularity = args.granularity
        costs = {
            key: getattr(args, key)
            for key in ("halocost", "tradecost", "ng")
            if getattr(args, key) is not None
        }
        part = get_partitioner(args.method, **costs)

        # balancing is all extents and connectivity, so no coordinate is read
        mb = pg.multiblock.BaseMultiBlock.from_grid(args.grid, quiet=False)
        before, maxbefore = len(mb.blocks), part.blkcells(mb).max()

        blkgroups = part.partition(mb, ranks, ranks_per_node, granularity)

        if granularity is not None:
            print(
                f"  {before} blocks -> {len(mb.blocks)} pieces,"
                f" largest {maxbefore} -> {part.blkcells(mb).max()}"
            )

        weights, edges = part.blkwts(mb), part.edges(mb)
        assign = np.empty(len(mb.blocks), dtype=np.int64)
        for r, group in enumerate(blkgroups):
            assign[group] = r
        load = np.bincount(assign, weights=weights, minlength=ranks)
        traffic = part.metrics(assign, weights, edges, ranks, ranks_per_node)

        efficiency = load.mean() / load.max() * 100
        maxblks = max(len(g) for g in blkgroups)
        nnodes = ranks // ranks_per_node

        writer = pg.writers.GridWriter(mb, args.grid, quiet=False)
        writer.add_partition(mb, blkgroups, ranks_per_node)

        print(
            f"Added a {ranks}x{ranks_per_node} partition"
            f" ({nnodes} node(s)) to {args.grid}\n\n",
            "Results of Load Balancing:\n",
            f"Total Number of blocks = {len(mb.blocks)}\n",
            f"Total Number of cells  = {int(part.blkcells(mb).sum())}\n",
            f"Total work = {int(weights.sum())} cells, with a halo {part.ng}"
            f" deep at {part.halocost} a cell, {part.tradecost}"
            f" more where traded\n\n",
            f"Maximum blocks on processor = {maxblks}\n",
            f"Maximum work on processor = {int(load.max())}\n",
            f"Efficiency = {efficiency}\n",
            f"Halo traffic of {traffic['total']} face cells:\n",
            f"  on a rank (no message) = {traffic['onrank']:.1f} %\n",
            f"  on a node (shared)    = {traffic['onnode']:.1f} %\n",
            f"  over the network      = {traffic['offnode']:.1f} %"
            f"  (busiest node {int(traffic['maxtrunk'])} cells)\n",
        )

        ceiling = part.ceiling(weights, ranks)
        if ceiling < 99.9:
            share = int(weights.sum() / ranks)
            biggest = int(weights.max())
            print(
                f" The largest block is {biggest} cells against an even share"
                f" of {share},\n so no placement of whole blocks can beat"
                f" {ceiling:.1f}%. Cut with -granularity to go further.\n"
            )

        if args.plot:
            # matplotlib only for a plot asked for
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.set_xlabel("Processor")
            ax.set_ylabel("Number of Cells")
            ax.plot(load, label="ncells", color="k")
            ax.set_ylim(bottom=0, top=None)

            ax1 = ax.twinx()
            ax1.set_ylabel("Number of Blocks")
            nblks = np.array([len(g) for g in blkgroups])
            ax1.plot(nblks, label="nblks", color="r")
            ax1.set_ylim(bottom=0, top=None)

            h1, la1 = ax.get_legend_handles_labels()
            h2, la2 = ax1.get_legend_handles_labels()
            ax.legend(h1 + h2, la1 + la2)
            plt.show()
