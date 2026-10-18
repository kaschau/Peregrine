import numpy as np

import peregrine as pg
from peregrine.tools.base import BaseTool


normalaxis = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
bigaxis = {0: "k", 1: "k", 2: "j"}
transposes = {
    "k": {"k": [1, 2, 4, 5], "j": [1, 4]},
    "j": {"k": [1, 2, 4, 5], "j": [1, 4]},
}


def faceplane(blk, nface):
    slices = {
        1: {"i": 0, "j": slice(None), "k": slice(None)},
        2: {"i": -1, "j": slice(None), "k": slice(None)},
        3: {"i": slice(None), "j": 0, "k": slice(None)},
        4: {"i": slice(None), "j": -1, "k": slice(None)},
        5: {"i": slice(None), "j": slice(None), "k": 0},
        6: {"i": slice(None), "j": slice(None), "k": -1},
    }
    s = slices[nface]
    plane = blk.nodes.get()[s["i"], s["j"], s["k"]]
    return (np.copy(plane[..., n]) for n in range(3))


def verify(mb):
    warn = False
    tol = 1e-7
    for blk in mb.blocks:
        for face in blk.intfaces:
            nface = face.nface
            nbr = face.nbr
            bc = face.intkind
            orient = face.orient
            bcname = face.bcname

            if nbr is None:
                assert bc != "interior" and face.rot is None, (
                    f"Block #{blk.nblki} face {nface} has no neighbor, "
                    f"but has kind {bc}"
                )
                assert orient is None, (
                    f"Block #{blk.nblki} face {nface} has no neighbor,"
                    f" but has orientation {orient}"
                )
                if bcname is None:
                    print(
                        f"Warning, block {blk.nblki} face {nface} is a"
                        " boundary with no name; a case cannot say what it is."
                    )
                    warn = True
                continue

            fx, fy, fz = faceplane(blk, face.nface)

            blk2 = mb.get_blk(nbr)
            nface2 = face.nbrnface
            orient2 = face.nbrorient

            if int(blk2.get_intface(nface2).nbr) != blk.nblki:
                raise ValueError(
                    f"Block {blk.nblki}'s face {nface} says it is connected"
                    f" to\nblock {blk2.nblki}'s face {nface2}, however block"
                    f" {blk2.nblki}'s\nface {nface2} says it is connected to a"
                    " different block."
                )

            fx2, fy2, fz2 = faceplane(blk2, nface2)

            orients = [
                int(i)
                for j, i in enumerate(orient2)
                if j != normalaxis[nface2]
            ]
            normal = [j for j in range(3) if j == normalaxis[nface2]][0]
            normal2 = [j for j in range(3) if j == normalaxis[nface]][0]

            big = bigaxis[normal]
            big2 = bigaxis[normal2]

            if orients[1] in transposes[big][big2]:
                fx = fx.T
                fy = fy.T
                fz = fz.T

            if orients[0] in [4, 5, 6]:
                fx = np.flip(fx, 0)
                fy = np.flip(fy, 0)
                fz = np.flip(fz, 0)

            if orients[1] in [4, 5, 6]:
                fx = np.flip(fx, 1)
                fy = np.flip(fy, 1)
                fz = np.flip(fz, 1)

            # move the interface onto its partner: the inverse of the transform a halo through it takes
            if face.rot is not None:
                shape = fx.shape
                points = np.column_stack((fx.ravel(), fy.ravel(), fz.ravel()))
                points = (points - face.xlate) @ face.rot
                fx = points[:, 0].reshape(shape)
                fy = points[:, 1].reshape(shape)
                fz = points[:, 2].reshape(shape)

            try:
                off_x = np.max(np.abs(fx - fx2))
                off_y = np.max(np.abs(fy - fy2))
                off_z = np.max(np.abs(fz - fz2))
            except ValueError:
                raise ValueError(
                    f"Error when comparing block {blk.nblki} and block"
                    f" {blk2.nblki} connection"
                )

            for axis, off in zip("xyz", (off_x, off_y, off_z)):
                if off > tol:
                    print(
                        f"Warning, the {axis} coordinates of face {nface} on"
                        f" block {blk.nblki} are not matching the {axis}"
                        f" coordinates of face {nface2} of block {blk2.nblki}"
                    )
                    print(f"Off by up to {off}")
                    warn = True

        # every block right handed
        nodes = blk.nodes.get()
        orig, xi = nodes[0, 0, 0], nodes[1, 0, 0]
        xj, xk = nodes[0, 1, 0], nodes[0, 0, 1]
        if np.dot(xk - orig, np.cross(xi - orig, xj - orig)) < 0.0:
            print(
                f"Warning, block {blk.nblki} is left handed."
                " This must be fixed."
            )
            warn = True

    return not warn


class VerifyTool(BaseTool):
    name = "verify"
    help = (
        "check a grid's connectivity and that joined faces' coordinates match"
    )

    def add_cli(self, parser):
        parser.add_argument("grid", help="the grid file")

    def main(self, args):
        mb = pg.multiblock.Grid.from_grid(args.grid, quiet=False)
        if not verify(mb):
            raise SystemExit(1)
        print("Grid is valid!")
