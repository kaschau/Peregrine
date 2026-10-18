import numpy as np

import peregrine as pg
from peregrine.partition import Conditioner
from peregrine.tools.verify import verify
from peregrine.tools.base import BaseTool


def _nextline(f):
    while True:
        line = f.readline().split()
        if line and not line[0].startswith("#"):
            return line


def read_pty(pty, nblks):
    with open(pty) as f:
        count = int(_nextline(f)[0])
        assert count == nblks, "property and connectivity disagree"
        ids = []
        for _ in range(nblks):
            line = _nextline(f)
            ids.append([line[n] for n in range(4, 15, 2)])
        while True:
            line = f.readline().split()
            if "2D" in line and "properties" in line:
                nprops = int(line[0])
                break
        names = {}
        for _ in range(nprops):
            line = f.readline().replace("(", "").replace(")", "").split()
            label = line[2].removeprefix("pdc:")
            names[line[0]] = None if label == "INTERBLK" else label
    return [[names[s] for s in surfaces] for surfaces in ids]


def read_conn(conn):
    with open(conn) as f:
        nblks = int(_nextline(f)[0])
        faces = []
        for _ in range(nblks):
            line = f.readline().split()[2:-1]
            pairs = [(int(line[4 * n + 2]), line[4 * n + 3]) for n in range(6)]
            nbrs = [b - 1 if b else None for b, _ in pairs]
            orients = [None if "0" in o else o for _, o in pairs]
            faces.append(list(zip(nbrs, orients)))
    return faces


def read_nodes(path, mb, binary, factor):
    with open(path, "rb" if binary else "r") as f:
        # past the header to the first block's extents
        while True:
            start = f.tell()
            line = f.readline()
            try:
                [int(b) for b in line.strip().split()]
                break
            except ValueError:
                pass
        for block in mb.blocks:
            f.seek(start)
            shape = tuple(int(b) for b in f.readline().strip().split())
            block.set_extents(*shape)
            count = shape[0] * shape[1] * shape[2]
            if binary:
                nodes = np.frombuffer(f.read(8 * count * 3), dtype=np.float64)
                nodes = nodes.reshape(shape + (3,))
                f.read(1)
            else:
                nodes = np.empty(shape + (3,))
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        for k in range(shape[2]):
                            line = f.readline().strip().split()
                            nodes[i, j, k] = [float(v) for v in line[0:3]]
            block.nodes.set(nodes * factor)
            start = f.tell()


class Gridpro2PgTool(BaseTool):
    name = "gridpro2pg"
    help = "translate a GridPro grid (blk.tmp, .conn, .pty) into a grid file"

    def add_cli(self, parser):
        parser.add_argument("blk", help="the GridPro block node file, blk.tmp")
        parser.add_argument(
            "conn", help="the GridPro connectivity file, blk.tmp.conn"
        )
        parser.add_argument(
            "pty", help="the GridPro property file, blk.tmp.pty"
        )
        parser.add_argument("out", help="the grid file to write")
        parser.add_argument(
            "-units",
            default="m",
            choices=("m", "in", "mm", "cm"),
            help="the grid's units",
        )
        parser.add_argument(
            "--binary",
            action="store_true",
            help="the node file is binary, not ascii",
        )

    def main(self, args):
        factor = {"m": 1.0, "in": 0.0254, "mm": 0.001, "cm": 0.01}[args.units]
        kind = "binary" if args.binary else "ascii"
        print(f"Reading the {kind} GridPro files")
        print(f"    {args.blk}\n    {args.conn}\n    {args.pty}\n")

        faces = read_conn(args.conn)
        nblks = len(faces)
        print(f"Found {nblks} blocks")
        names = read_pty(args.pty, nblks)

        mb = pg.multiblock.Grid()
        for _ in range(nblks):
            mb.add_blk()
        for blk, conns, labels in zip(mb.blocks, faces, names):
            for face, (nbr, orient), label in zip(blk.intfaces, conns, labels):
                if nbr is None:
                    # GridPro's label is the name the case binds values to
                    face.bcname = label
                else:
                    # interior or periodic: mark_periodics tells them apart below
                    face.set_interior(nbr, orient)
        read_nodes(args.blk, mb, args.binary, factor)

        # a periodic arrives joined like an interior interface; the points say which are and their transform, which verify needs
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
        print("GridPro to PEREGRINE translation done.")
