from pathlib import Path

import numpy as np

import peregrine as pg
from peregrine.partition import Conditioner
from peregrine.tools.verify import verify
from peregrine.tools.base import BaseTool


facemap = {
    "small_i": 1,
    "large_i": 2,
    "small_j": 3,
    "large_j": 4,
    "small_k": 5,
    "large_k": 6,
}
orientmap = {"i": 1, "j": 2, "k": 3, "-i": 4, "-j": 5, "-k": 6}


def read_tlns3d(fname, mb, factor):
    # scipy is a tool's dependency, not the solver's
    from scipy.io import FortranFile

    with FortranFile(fname, "r") as f90:
        nblks = f90.read_ints(dtype=np.int32)[0]
        for _ in range(nblks):
            mb.add_blk()
        nijks = np.array_split(f90.read_ints(dtype=np.int32), nblks)
        for nijk, blk in zip(nijks, mb.blocks):
            ni, nj, nk = nijk
            nodes = f90.read_reals(dtype=np.float64)
            nodes = nodes.reshape((ni, nj, nk, 3), order="F")
            blk.set_extents(ni, nj, nk)
            blk.nodes.set(nodes * factor)


def read_doms(here, mb, factor):
    nblks = len(list(here.glob("info.dom*")))
    print(f"Reading {nblks} ICEM domain files")
    for _ in range(nblks):
        mb.add_blk()
    for blk in mb.blocks:
        fname = here / f"info.dom{blk.nblki}"
        with open(fname) as f:
            _, ni, nj, nk = (int(v) for v in f.readline().strip().split()[:4])
        points = np.genfromtxt(fname, comments="domain.")
        blk.set_extents(ni, nj, nk)
        blk.nodes.set(np.reshape(points[:, 0:3], (ni, nj, nk, 3)) * factor)


def axesof(line):
    return [d.replace("-", "") for d in line[2:5]]


def faceof(dirs, mins, maxs):
    for lo, hi, direction in zip(mins, maxs, dirs):
        if lo == hi:
            side = "small" if lo == "1" else "large"
            return facemap[f"{side}_{direction}"]
    return None


def orientof(line, adjline, nface):
    if line[2:5] == adjline[2:5]:
        return "123"
    orient = {}
    for n in range(3):
        curr, adjc = line[2 + n], adjline[2 + n]
        # the neighbor's sign flips on the normal axis at a large end, and on a tangential axis this block lists as negative
        side = "small" if curr.startswith("-") else "large"
        onface = facemap[f"{side}_{curr[-1]}"] == nface
        if onface != curr.startswith("-"):
            negative = adjc.startswith("-")
            adjc = adjc.replace("-", "") if negative else f"-{adjc}"
        orient[curr.replace("-", "")] = adjc
    return "".join(str(orientmap[orient[a]]) for a in "ijk")


# the topology format: https://support.ansys.com/staticassets/ANSYS/Initial%20Content%20Entry/General%20Articles%20-%20Products/ICEM%20CFD%20Interfaces/multiblock.htm
def read_topo(topo, mb):
    lines = Path(topo).read_text().splitlines(keepends=True)

    # the tagged boundary interfaces: the tag is the interface's name
    blk = None
    for raw in lines:
        if raw.startswith("# Boundary conditions and/or"):
            blk = mb.get_blk(int(raw.strip().split(".")[-1]) - 1)
            continue
        if raw == "\n":
            blk = None
        if blk is None:
            continue
        line = raw.strip().split()
        if line[1] != "f" or line[0] == "DEFAULT_SUBFACE":
            continue
        nface = faceof("ijk", line[2:5], line[5:8])
        blk.get_intface(nface).bcname = line[0]

    # the periodics, joined like interior connections; mark_periodics reads their transform
    it = iter(lines)
    for raw in it:
        if not raw.startswith("# Periodic info for domain"):
            continue
        current = int(raw.strip().split(".")[-1]) - 1
        blk = mb.get_blk(current)
        line = next(it).replace("-", " -").strip().split()
        if not line:
            continue
        adjline = next(it).replace("-", " -").strip().split()
        adjacent = int(adjline[1].split(".")[-1]) - 1
        nface = faceof(axesof(line), line[6:9], line[9:12])
        if line[5] != "f":
            raise ValueError(
                f"expected a face to set face {nface} of block {current},"
                f" not a {line[5]}"
            )
        orient = orientof(line, adjline, nface)
        blk.get_intface(nface).set_interior(adjacent, orient)
        # a block periodic with itself is listed once, so its opposite interface is set here too
        if current == adjacent:
            opp = nface + 1 if nface % 2 else nface - 1
            blk.get_intface(opp).set_interior(current, orient)

    # the interior connections, in the order of the interfaces still unassigned
    it = iter(lines)
    for raw in it:
        if not raw.startswith("# Connectivity for domain"):
            continue
        current = int(raw.strip().split(".")[-1]) - 1
        blk = mb.get_blk(current)
        blank = [f for f in blk.intfaces if f.bcname is None and f.nbr is None]
        unassigned = [f.nface for f in blank]
        for nface in unassigned:
            line = next(it).replace("-", " -").strip().split()
            adjline = next(it).replace("-", " -").strip().split()
            found = faceof(axesof(line), line[6:9], line[9:12])
            if found != nface:
                raise ValueError(
                    f"the connectivity of block {current} gives face {found}"
                    f" where face {nface} was expected"
                )
            if line[5] != "f":
                raise ValueError(
                    f"expected a face to set face {nface} of block {current},"
                    f" not a {line[5]}"
                )
            adjacent = int(adjline[1].strip().split(".")[-1]) - 1
            orient = orientof(line, adjline, nface)
            blk.get_intface(nface).set_interior(adjacent, orient)


class Icem2PgTool(BaseTool):
    name = "icem2pg"
    help = "translate an ICEM Multi-Block Info grid into a grid file"

    def add_cli(self, parser):
        parser.add_argument("topo", help="the topology file, info.topo")
        parser.add_argument("out", help="the grid file to write")
        parser.add_argument(
            "-nodes",
            help="the TLNS3D-MB node file, double precision; without it the "
            "nodes are read from the single precision info.dom files beside "
            "the topology",
        )
        parser.add_argument(
            "-units",
            default="m",
            choices=("m", "in", "mm", "cm"),
            help="the grid's units",
        )

    def main(self, args):
        factor = {"m": 1.0, "in": 0.0254, "mm": 0.001, "cm": 0.01}[args.units]
        mb = pg.multiblock.Grid()
        if args.nodes:
            read_tlns3d(args.nodes, mb, factor)
        else:
            read_doms(Path(args.topo).parent, mb, factor)
        read_topo(args.topo, mb)

        found = mb.mark_periodics()
        for kind, n in sorted(found.items()):
            print(f"Found {n} {kind} face(s) among the interfaces")

        # a grid that does not verify is a warning; conditioning breaking one that did is an error
        verified = verify(mb)
        if not verified:
            print("  NOTE: the translated grid does not verify, see above")
        Conditioner().condition(mb)
        if verified and not verify(mb):
            raise ValueError("Conditioning invalidated the grid.")

        print(f"Writing the {len(mb.blocks)} block grid to {args.out}")
        pg.writers.GridWriter(mb, args.out, quiet=False).write(mb)
        print("ICEM to PEREGRINE translation done.")
