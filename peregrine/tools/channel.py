from pathlib import Path

import numpy as np
import yaml

import peregrine as pg
from peregrine.tools.base import BaseTool


template = """---
domain: channel        # channel, or bl for a boundary layer

# Geometry
delta: 0.05                # channel half height or boundary layer thickness, m
z-dim: 3                  # width, in delta
y-dim: 2                 # height, in delta; a channel is two delta high
lx: 6                 # length, in delta

# Flow
viscosity: 1.81e-5         # kinematic viscosity, m^2/s
retau: 587.19              # friction Reynolds number

# Resolution, in wall units; a default is the value in the comment
nvsl: default              # points across the viscous sublayer, y+ < 5 (6)
gr-buff: default            # growth rate through the buffer layer (1.1)
gr-log: default             # growth rate through the log layer (1.1)
dy-core: default        # wall-normal spacing in the core or free stream (20)
dx: default            # streamwise spacing (20)
dz: default            # spanwise spacing (20)

# Boundaries: which axes repeat, and the name in bcvals of each face that
# does not, numbered as the mesher numbers them: 1 and 2 the x faces, 3 and
# 4 the y faces, 5 and 6 the z faces
periodic: [true, false, true]
bcnames: {3: wall, 4: wall}
"""

defaults = {
    "nvsl": 6,
    "gr-buff": 1.1,
    "gr-log": 1.1,
    "dy-core": 20.0,
    "dx": 20.0,
    "dz": 20.0,
}


def grow_span(start, stop, gr, dy0):
    assert stop > start > 0.0 and gr > 1.0
    pts = [start]
    dy = dy0 * gr
    while pts[-1] < stop:
        pts.append(pts[-1] + dy)
        dy *= gr
    if abs(pts[-1] - stop) > abs(pts[-2] - stop):
        pts = pts[0:-1]
    pts = np.array(pts)
    return start + (pts - start) / (stop - start) * stop


def grow_dy(dy0, dy1, gr):
    assert 0.0 < dy0 < dy1 and gr > 1.0
    pts = [0]
    dy = dy0 * gr
    while dy < dy1:
        pts.append(pts[-1] + dy)
        dy *= gr
    return np.array(pts)


def wallnormal(inp):
    def setting(key):
        given, default = inp[key], defaults[key]
        return default if given == "default" else type(default)(given)

    nVSL = setting("nvsl")
    assert nVSL > 2
    yvsl = np.linspace(0, 5, nVSL)
    ybuff = grow_span(5, 30, setting("gr-buff"), yvsl[-1] - yvsl[-2])
    dYplusCore = setting("dy-core")
    grlog = setting("gr-log")
    ylog = ybuff[-1] + grow_dy(ybuff[-1] - ybuff[-2], dYplusCore, grlog)

    delta, viscosity = float(inp["delta"]), float(inp["viscosity"])
    ReTau = float(inp["retau"])
    yp1 = viscosity / (ReTau * viscosity / delta)
    domain = inp["domain"]
    if domain == "channel":
        yend = delta / yp1
    elif domain == "bl":
        yend = float(inp["y-dim"]) * delta / yp1
    else:
        raise ValueError(f"domain is channel or bl, not {domain!r}")
    ncore = int((yend - ylog[-1]) / dYplusCore)
    ycore = np.linspace(ylog[-1], yend, ncore)

    ys = np.concatenate((yvsl, ybuff[1:-1], ylog[0:-1], ycore))
    if domain == "channel":
        ys = np.concatenate((ys, 2 * ys[-1] - np.flip(ys)[1::]))
    return ys, yp1, (len(yvsl) - 1, len(ybuff) - 1)


class ChannelTool(BaseTool):
    name = "channel"
    help = "make a channel or boundary layer grid, spaced in wall units"

    def add_cli(self, parser):
        parser.add_argument(
            "spec", nargs="?", help="the yaml describing the grid"
        )
        parser.add_argument("out", nargs="?", help="the grid file to write")
        parser.add_argument(
            "-template",
            metavar="FILE",
            help="write a template yaml to FILE and stop",
        )

    def main(self, args):
        if args.template:
            Path(args.template).write_text(template)
            return
        if not (args.spec and args.out):
            raise SystemExit(
                "peregrine channel: the yaml and the grid file to write"
            )
        with open(args.spec) as f:
            inp = yaml.safe_load(f)

        ys, yp1, (nVSL, nbuff) = wallnormal(inp)
        delta = float(inp["delta"])
        dXplus = defaults["dx"] if inp["dx"] == "default" else float(inp["dx"])
        dZplus = defaults["dz"] if inp["dz"] == "default" else float(inp["dz"])
        lx, lz = float(inp["lx"]) * delta, float(inp["z-dim"]) * delta
        ly = ys[-1] * yp1
        dx, dz = dXplus * yp1, dZplus * yp1
        nx, ny, nz = int(lx / dx), len(ys), int(lz / dz)

        grid = pg.multiblock.Grid()
        periodic = [bool(p) for p in inp["periodic"]]
        bcnames = {int(face): str(name) for face, name in inp["bcnames"].items()}
        box = pg.mesher.CubeMesher(lengths=[lx, ly, lz], blkdims=[nx, ny, nz], periodic=periodic, bcnames=bcnames)
        box.fill(grid)
        # the box's even spacing in y replaced by the wall-unit spacing
        blk = grid.blocks[0]
        nodes = blk.nodes.get()
        nodes[..., 1] = ys[np.newaxis, :, np.newaxis] * yp1
        blk.nodes.set(nodes)

        inlog = np.flatnonzero((ys > 30.0) & (ys * yp1 / delta < 0.2))
        print(
            "Summary:\n"
            f"Domain type: {inp['domain']}\n"
            f"{nx=}, {ny=}, {nz=}\n"
            f"Total Cells: {(nx - 1) * (ny - 1) * (nz - 1)}\n"
            f"Viscous Sub Layer Cells: {nVSL}\n"
            f"Buffer Layer Cells: {nbuff}\n"
            f"Log Layer Cells: {len(inlog)}\n"
        )
        pg.writers.GridWriter(grid, args.out, quiet=False).write(grid)
