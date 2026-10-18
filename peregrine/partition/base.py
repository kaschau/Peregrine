import numpy as np

from peregrine.partition.cut import CutMixin


class BasePartitioner(CutMixin):
    name = None

    def __init__(
        self,
        tol=0.05,
        npasses=10,
        nstarts=6,
        halocost=0.25,
        tradecost=0.06,
        ng=1,
    ):
        # how far past an even share a rank may be loaded
        self.tol = tol
        self.npasses = npasses
        self.nstarts = nstarts
        # what a halo cell costs a rank, in interior cells, behind any interface
        self.halocost = halocost
        # what a traded halo cell costs besides: its pack, unpack and fill
        self.tradecost = tradecost
        # the case's halo depth, which the grid does not know
        self.ng = ng

    def partition(self, mb, nranks, ranks_per_node, granularity=None):
        if nranks % ranks_per_node:
            raise ValueError(
                f"{nranks} ranks is not a whole number of nodes at"
                f" {ranks_per_node} ranks per node."
            )

        if granularity is not None:
            maxcells = int(self.blkcells(mb).sum() / nranks / granularity)
            self.cut_down(mb, maxcells)

        weights, edges = self.blkwts(mb), self.edges(mb)
        nnodes = nranks // ranks_per_node
        assign = self.bynode(weights, edges, nnodes, ranks_per_node)

        return [np.flatnonzero(assign == r).tolist() for r in range(nranks)]

    def _assign(self, weights, edges, k):
        raise NotImplementedError

    ###########################################################################
    # The grid as a weighted graph
    ###########################################################################
    @staticmethod
    def blkcells(mb):
        return np.array([b.ncells for b in mb.blocks])

    def blkwts(self, mb):
        weights = self.blkcells(mb).astype(np.float64)
        for blk, face in mb.intfaces():
            halo = self.ng * self.planecells(blk, face)
            trade = self.tradecost if face.nbr is not None else 0.0
            weights[blk.nblki] += (self.halocost + trade) * halo
        return weights

    @staticmethod
    def planecells(blk, face):
        cells = [blk.ni - 1, blk.nj - 1, blk.nk - 1]
        del cells[face.axis]
        return cells[0] * cells[1]

    def edges(self, mb):
        edges = {}
        for blk, face in mb.intfaces():
            if face.nbr is None:
                continue
            key = tuple(sorted((blk.nblki, face.nbr)))
            # a connection shows up from both blocks, count it once
            if blk.nblki > face.nbr and key in edges:
                continue
            edges[key] = edges.get(key, 0) + self.planecells(blk, face)
        return edges

    @staticmethod
    def traffic(assign, edges):
        return sum(w for (a, b), w in edges.items() if assign[a] != assign[b])

    ###########################################################################
    # Cutting a grid down until it can be balanced
    ###########################################################################
    def cut_down(self, mb, maxcells):
        while True:
            sizes = self.blkcells(mb)
            worst = int(sizes.argmax())
            if sizes[worst] <= maxcells:
                return
            blk = mb.blocks[worst]

            best = None
            for axis, n in zip("ijk", (blk.ni, blk.nj, blk.nk)):
                # every piece needs a cell, so n nodes take at most n-2 cuts
                ncuts = min(int(np.ceil(sizes[worst] / maxcells)) - 1, n - 2)
                if ncuts < 1:
                    continue
                # a cut runs on through every block its plane meets, so the cheapest axis is the shortest path
                cost = len(self.cutpath(mb, blk.nblki, axis)) * ncuts
                if best is None or cost < best[0]:
                    best = (cost, axis, ncuts)
            if best is None:
                raise ValueError(
                    f"block {blk.nblki} holds {sizes[worst]} cells and cannot be cut"
                    f" under {maxcells}: it is only {blk.ni}x{blk.nj}x{blk.nk} nodes"
                )
            self.cut(mb, [[blk.nblki, best[1], best[2]]])

    @staticmethod
    def ceiling(sizes, nranks):
        return min(1.0, np.sum(sizes) / nranks / np.max(sizes)) * 100

    ###########################################################################
    # Placing the graph
    ###########################################################################
    def bynode(self, weights, edges, nnodes, ranks_per_node):
        if nnodes <= 1:
            return self._assign(weights, edges, ranks_per_node)

        nodeof = self._assign(weights, edges, nnodes)
        assign = np.full(weights.shape[0], -1, dtype=np.int64)
        for node in range(nnodes):
            blks = np.where(nodeof == node)[0]
            if len(blks) == 0:
                continue
            subw, sube = self.subgraph(blks, weights, edges)
            localrank = self._assign(subw, sube, ranks_per_node)
            assign[blks] = node * ranks_per_node + localrank
        nranks = nnodes * ranks_per_node
        return self.rebalance(assign, weights, edges, nranks, ranks_per_node)

    @staticmethod
    def subgraph(blks, weights, edges):
        local = {int(b): i for i, b in enumerate(blks)}
        sube = {}
        for (a, b), w in edges.items():
            if a in local and b in local:
                sube[(local[a], local[b])] = w
        return weights[blks], sube

    # runs after both levels of the split: each carries its own slack and the step time sees the product
    def rebalance(self, assign, weights, edges, nranks, ranks_per_node):
        assign = assign.copy()
        load = np.bincount(assign, weights=weights, minlength=nranks)
        cap = load.mean() * (1.0 + self.tol)
        aff = {}
        for (a, b), w in edges.items():
            aff.setdefault(a, []).append((b, w))
            aff.setdefault(b, []).append((a, w))

        for _ in range(len(weights)):
            hi = int(np.argmax(load))
            if load[hi] <= cap:
                break
            best = None
            for p in np.where(assign == hi)[0]:
                w = weights[p]
                talk = {}
                for nbr, e in aff.get(int(p), []):
                    talk[int(assign[nbr])] = talk.get(int(assign[nbr]), 0) + e
                for r in range(nranks):
                    if r == hi or load[r] + w >= load[hi]:
                        continue
                    offnode = r // ranks_per_node != hi // ranks_per_node
                    key = (offnode, load[r] + w, -talk.get(r, 0))
                    if best is None or key < best[0]:
                        best = (key, int(p), r)
            if best is not None:
                _, p, r = best
                assign[p] = r
                load[hi] -= weights[p]
                load[r] += weights[p]
                continue

            # no single move lowers the max, so swap for something lighter
            for p in np.where(assign == hi)[0]:
                w = weights[p]
                lighter = np.where((weights < w) & (assign != hi))[0]
                r = assign[lighter]
                newr = load[r] - weights[lighter] + w
                newhi = load[hi] - w + weights[lighter]
                ok = (newr < load[hi]) & (newhi < load[hi])
                for n in np.flatnonzero(ok):
                    q = lighter[n]
                    offnode = r[n] // ranks_per_node != hi // ranks_per_node
                    key = (offnode, max(newr[n], newhi[n]))
                    if best is None or key < best[0]:
                        best = (key, int(p), int(q))
            if best is None:
                break
            _, p, q = best
            r = assign[q]
            assign[p], assign[q] = r, hi
            load[hi] += weights[q] - weights[p]
            load[r] += weights[p] - weights[q]
        return assign

    def metrics(self, assign, weights, edges, k, ranks_per_node):
        total = sum(edges.values())
        rpn = ranks_per_node
        n = (k + rpn - 1) // rpn

        cut = onnode = offnode = 0
        trunk = np.zeros(n)
        for (a, b), w in edges.items():
            ra, rb = int(assign[a]), int(assign[b])
            if ra == rb:
                continue
            cut += w
            na, nb = ra // rpn, rb // rpn
            if na == nb:
                onnode += w
            else:
                offnode += w
                trunk[na] += w
                trunk[nb] += w

        return {
            "total": total,
            "onrank": 100.0 * (1.0 - cut / total) if total else 100.0,
            "onnode": 100.0 * onnode / total if total else 0.0,
            "offnode": 100.0 * offnode / total if total else 0.0,
            # the busiest node's network traffic is what paces a synchronous step
            "maxtrunk": float(trunk.max()) if n else 0.0,
        }
