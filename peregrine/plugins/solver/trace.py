from pathlib import Path

import numpy as np

from peregrine.launch.kernel import Kernel
from peregrine.launch.ranges import CellCenterRange
from peregrine.misc import get_comm_rank_size
from peregrine.plugins.solver.base import BaseSolverPlugin
from peregrine.plugins.solver.cadence import Cadence


def point(spec):
    return np.array([spec["p0"]], float)


def line(spec):
    p0, p1 = (np.array(spec[k], float) for k in ("p0", "p1"))
    return p0 + np.linspace(0.0, 1.0, spec["n"])[:, None] * (p1 - p0)


def plane(spec):
    p0, p1, p2 = (np.array(spec[k], float) for k in ("p0", "p1", "p2"))
    a = np.linspace(0.0, 1.0, spec["n01"])[:, None, None]
    b = np.linspace(0.0, 1.0, spec["n02"])[None, :, None]
    return (p0 + a * (p1 - p0) + b * (p2 - p0)).reshape(-1, 3)


makers = {"point": point, "line": line, "plane": plane}


class Trace(BaseSolverPlugin):
    name = "trace"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        self.specs = cfgsect["traces"]
        for tag, spec in self.specs.items():
            if spec.get("type") not in makers:
                raise ValueError(f"trace {tag}: type is point, line or plane")
            if "file" not in spec:
                raise ValueError(f"trace {tag}: names the file it writes")
        own = {"niter-out", "dt-out"}
        self.cadences = {
            tag: Cadence(spec) if own & set(spec) else self.cadence
            for tag, spec in self.specs.items()
        }

    def arrays(self, solver):
        return {"tracelist": dict(kind=CellCenterRange, dtype=np.int32), "traced": dict(kind=CellCenterRange)}

    def due(self, solver):
        return any(cadence.due(solver) for cadence in self.cadences.values())

    def tnext(self, solver):
        times = [c.next(solver) for c in self.cadences.values()]
        return min((t for t in times if t is not None), default=None)

    def start(self, solver):
        comm, rank, size = get_comm_rank_size()
        # per trace: its file and its points; per block: its traced cells as (trace, point, i, j, k, center), in slot order
        self.traces, self.slots = {}, {}
        for tag, spec in self.specs.items():
            points = makers[spec["type"]](spec)
            self.traces[tag] = (spec["file"], points)
            for n, nblki, i, j, k, center in self._nearest(solver, points, comm):
                self.slots.setdefault(nblki, []).append((tag, n, i, j, k, center))
            if rank == 0 and not Path(spec["file"]).exists():
                names = solver.sim.iovars.privars
                header = "time, x, y, z, " + ", ".join(names) + "\n"
                Path(spec["file"]).write_text(header)
        self.nvars = len(solver.sim.iovars.privars)
        for nblki, cells in self.slots.items():
            self._write_list(solver.get_blk(nblki), cells, solver.backend.order)
        tpl = f"{__package__}.kernels.trace"
        self.kernel = Kernel(solver.provider, tpl, Q=solver.regs[0], state="state", tracelist="tracelist", traced="traced")

    def _write_list(self, blk, cells, order):
        n0, n1, n2 = blk.tracelist.shape[:3]
        count = len(cells)
        if count * self.nvars > n0 * n1 * n2:
            msg = f"trace: block {blk.nblki} holds {count} traced points, more rows than its cells carry"
            raise ValueError(msg)
        flat = np.zeros(n0 * n1 * n2, dtype=np.int32)
        # the listed layout: the count of every listed cell, then its one segment's count, then each cell as a code on the allocation's extents
        flat[0] = flat[1] = count
        flat[2 : 2 + count] = [i + n0 * (j + n1 * k) for _, _, i, j, k, _ in cells]
        blk.tracelist.set(flat.reshape(blk.tracelist.shape, order=order))

    @staticmethod
    def _nearest(solver, points, comm):
        best = np.full(len(points), np.inf)
        where = np.zeros((len(points), 4), dtype=int)
        centers = np.zeros((len(points), 3))
        for blk in solver.blocks:
            ng = blk.ng
            cells = blk.xcc.get()[blk.interior]
            shape, flat = cells.shape[:3], cells.reshape(-1, 3)
            # |c - p|^2 as |c|^2 - 2 c.p + |p|^2: one matvec per point
            cc = (flat**2).sum(axis=1)
            for n, p in enumerate(points):
                d = cc - 2.0 * (flat @ p) + p @ p
                m = d.argmin()
                if d[m] < best[n]:
                    best[n] = d[m]
                    i, j, k = np.unravel_index(m, shape)
                    where[n] = (blk.nblki, i + ng, j + ng, k + ng)
                    centers[n] = flat[m]
        # the rank whose cell is nearest keeps the point
        everyone = np.array(comm.allgather(best))
        mine = np.flatnonzero(everyone.argmin(axis=0) == comm.rank)
        return [(int(n), *(int(x) for x in where[n]), centers[n]) for n in mine]

    def __call__(self, solver):
        comm, rank, size = get_comm_rank_size()
        due = [tag for tag in self.traces if self.cadences[tag].due(solver)]
        if not due:
            return
        solver.launch(self.kernel, "full")
        rows = {tag: [] for tag in due}
        for nblki, cells in self.slots.items():
            values = solver.get_blk(nblki).traced.head(len(cells) * self.nvars).reshape(len(cells), self.nvars)
            for slot, (tag, n, _, _, _, center) in enumerate(cells):
                if tag in rows:
                    rows[tag].append((n, [*center, *values[slot]]))
        everyone = comm.gather(rows, root=0)
        if rank != 0:
            return
        tme = solver.tme.get()
        for tag in due:
            fname = self.traces[tag][0]
            theirs = sorted(r for ranks in everyone for r in ranks[tag])
            with open(fname, "a") as f:
                for n, values in theirs:
                    text = ", ".join(f"{v:.8e}" for v in (tme, *values))
                    f.write(text + "\n")
