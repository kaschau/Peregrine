import numpy as np

import peregrine as pg
from peregrine.multiblock.interface import BaseInterface
from peregrine.tools.base import BaseTool


def turnof(face, axis):
    R = face.rot
    sx, sy, sz = R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]
    sine = 0.5 * (sx * axis[0] + sy * axis[1] + sz * axis[2])
    cosine = 0.5 * (np.trace(R) - 1.0)
    return np.degrees(np.arctan2(sine, cosine))


class RotateTool(BaseTool):
    name = "rotate"
    help = "copy a sector grid about its axis into so many sectors"

    def add_cli(self, parser):
        parser.add_argument("grid", help="the sector's grid file")
        parser.add_argument("out", help="the grid file to write")
        parser.add_argument(
            "-segments",
            type=int,
            required=True,
            help="how many sectors to end with",
        )
        parser.add_argument(
            "-angle",
            type=float,
            required=True,
            help="the sector's angle, degrees",
        )
        parser.add_argument(
            "-axis",
            required=True,
            help="the axis the sector turns about, x,y,z",
        )

    def main(self, args):
        nseg, angle = args.segments, args.angle
        if nseg <= 1:
            raise SystemExit(
                "peregrine rotate: -segments is the sectors to end with,"
                " above one"
            )
        axis = np.array([float(a) for a in args.axis.split(",")])
        axis /= np.linalg.norm(axis)
        closed = abs(nseg * angle - 360.0) < 1e-10

        from_grid = pg.multiblock.Grid.from_grid(args.grid, quiet=False)
        nblks = len(from_grid.blocks)
        # a halo through the sector's high interface is turned by +angle, through its low one by -angle
        high = {}
        for blk, face in from_grid.intfaces():
            if face.rot is None:
                continue
            turn = turnof(face, axis)
            if abs(abs(turn) - angle) > 1e-6:
                raise ValueError(
                    f"block {blk.nblki} face {face.nface} is periodic by"
                    f" {turn:.6f} degrees, not the sector's {angle}"
                )
            high[(blk.nblki, face.nface)] = turn > 0.0
        if not high:
            raise ValueError(
                f"{args.grid} has no periodic faces to turn about"
            )

        tgt = pg.multiblock.Grid()
        for _ in range(nblks * nseg):
            tgt.add_blk()
        whole = BaseInterface.rotabout(axis, nseg * angle)
        for k in range(nseg):
            R = BaseInterface.rotabout(axis, k * angle)
            for blk in from_grid.blocks:
                tblk = tgt.blocks[k * nblks + blk.nblki]
                tblk.set_extents(blk.ni, blk.nj, blk.nk)
                nodes = blk.nodes.get()
                turned = (nodes.reshape(-1, 3) @ R.T).reshape(nodes.shape)
                tblk.nodes.set(turned)
                for tface, face in zip(tblk.intfaces, blk.intfaces):
                    if face.nbr is None:
                        tface.copy_from(face)
                        continue
                    nbr, orient = face.nbr, face.orient
                    if face.rot is None:
                        tface.set_interior(nbr + k * nblks, orient)
                        continue
                    # a periodic joins the next sector round, or spans the whole sweep at the ends
                    step = 1 if high[(blk.nblki, face.nface)] else -1
                    partner = k + step
                    across = nbr + (partner % nseg) * nblks
                    if 0 <= partner < nseg:
                        tface.set_interior(nbr + partner * nblks, orient)
                    elif closed:
                        tface.set_interior(across, orient)
                    else:
                        tface.copy_from(face)
                        tface.nbr = across
                        tface.set_periodic(rot=whole if step > 0 else whole.T)

        print(f"Writing the {len(tgt.blocks)} block grid to {args.out}")
        pg.writers.GridWriter(tgt, args.out, quiet=False).write(tgt)
