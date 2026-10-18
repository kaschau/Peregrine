import numpy as np

from peregrine.misc import Progress


class BaseInterpolator:
    name = None

    def __init__(self, deep_search=False):
        self.deep_search = deep_search

    # whether a kind can give a value outside the range it was given, and so wants clipping to it
    overshoots = True

    def mapper(self, spts, tpts):
        raise NotImplementedError

    def interpolate(self, src, tgt):
        bounds = self.find_bounds(tgt, src)
        with Progress(len(tgt.blocks)) as bar:
            for tblk, sblks in zip(tgt.blocks, bounds):
                self.fill_blk(sblks, tblk)
                bar.step(f"Interpolating block {tblk.nblki}")

    def fill_blk(self, sblks, tblk):
        # the points are the same for every variable, so the geometry is worked out once
        centers = [blk.xcc.view().reshape(-1, 3) for blk in sblks]
        spts = np.concatenate(centers)
        tpts = tblk.xcc.view().reshape(-1, 3)
        onto = self.mapper(spts, tpts)

        shape = tblk.q.shape[:3]
        # written in place through view(), which is the array itself only on the host backend
        tprims = tblk.q.view()
        sprims = [blk.q.view() for blk in sblks]
        for i in range(sprims[0].shape[-1]):
            svals = np.concatenate([q[..., i].ravel() for q in sprims])
            tvals = onto(svals)

            if self.overshoots:
                tvals = np.clip(tvals, np.min(svals), np.max(svals))

            tprims[:, :, :, i] = tvals.reshape(shape)

    def find_bounds(self, tgt, src):
        src.make_metrics()
        tgt.make_metrics()

        # each from block's box and hull, the same for every block being filled
        sboxes = {}
        for blk in src.blocks:
            nodes = blk.nodes.view()
            sboxes[blk.nblki] = np.stack([nodes.min(axis=(0, 1, 2)), nodes.max(axis=(0, 1, 2))], axis=-1)
        shulls = {blk.nblki: self.hull(blk) for blk in src.blocks}

        found = []
        bar = Progress(len(tgt.blocks))
        for tblk in tgt.blocks:
            centers = tblk.xcc.view().reshape(-1, 3)
            tlo, thi = centers.min(axis=0), centers.max(axis=0)

            covered = np.zeros(len(centers), dtype=bool)
            inside = []
            for sblk in src.blocks:
                lo, hi = sboxes[sblk.nblki].T
                # a block whose box does not reach this one holds none of its cells
                if np.any(lo > thi) or np.any(hi < tlo):
                    continue

                hits = self.in_hull(shulls[sblk.nblki], centers)
                if hits.any():
                    inside.append(sblk)
                    covered |= hits
                # a deep search keeps looking once every cell is covered, for grids that overlap in more places than the first hit
                if not self.deep_search and covered.all():
                    break

            if not inside:
                raise ValueError(
                    f"block {tblk.nblki} of the grid being interpolated onto lies"
                    " entirely outside the grid being interpolated from"
                )
            found.append(inside)
            bar.step(f"Finding block {tblk.nblki} bounds")

        return found

    @staticmethod
    def hull(blk):
        nodes = blk.nodes.view()
        # every surface node: a hull of the corners and edge midpoints is wrong as soon as the block curves
        surface = np.zeros(nodes.shape[:3], dtype=bool)
        surface[[0, -1]] = True
        surface[:, [0, -1]] = True
        surface[:, :, [0, -1]] = True
        # scipy is a tool's dependency, not the solver's
        from scipy import spatial

        return spatial.ConvexHull(nodes[surface])

    @staticmethod
    def in_hull(hull, pts):
        # a point on a facet counts as in, so allow the rounding that puts it a hair outside
        tol = 1e-12 * np.linalg.norm(hull.max_bound - hull.min_bound)
        dist = pts @ hull.equations[:, :3].T + hull.equations[:, 3]
        return np.all(dist <= tol, axis=1)
