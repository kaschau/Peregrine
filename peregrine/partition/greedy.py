import numpy as np

from peregrine.partition.base import BasePartitioner


class GreedyPartitioner(BasePartitioner):
    name = "greedy"

    def _assign(self, weights, edges, k):
        if k <= 1:
            return np.zeros(weights.shape[0], dtype=np.int64)

        nblks = weights.shape[0]
        # fewer blocks than ranks leaves ranks empty; only cutting can fill them
        if nblks <= k:
            return np.arange(nblks, dtype=np.int64)

        order = np.argsort(weights, kind="stable")[::-1]
        # one start at least, and no more than there are blocks to seed from
        nstarts = max(1, min(self.nstarts, nblks))
        stride = max(1, nblks // nstarts)
        picks = [s * stride % nblks for s in range(1, nstarts)]
        seeds = [None] + order[picks].tolist()

        best, bestkey = None, None
        for seed in seeds:
            assign = self._once(weights, edges, k, seed)
            load = np.bincount(assign, weights=weights, minlength=k)
            key = (self.traffic(assign, edges), load.max() - load.min())
            if bestkey is None or key < bestkey:
                best, bestkey = assign, key
        return best

    def _once(self, weights, edges, k, seed):
        nblks = weights.shape[0]
        target = weights.sum() / k
        cap = target * (1.0 + self.tol)

        nbrs = [[] for _ in range(nblks)]
        for (a, b), w in edges.items():
            nbrs[a].append((b, w))
            nbrs[b].append((a, w))

        assign = np.full(nblks, -1, dtype=np.int64)
        load = np.zeros(k)
        # edge weight from each block to each part
        aff = np.zeros((nblks, k))
        order = np.argsort(weights, kind="stable")[::-1]

        def place(b, r):
            assign[b] = r
            load[r] += weights[b]
            for nbr, w in nbrs[b]:
                aff[nbr, r] += w

        def unplace(b):
            r = assign[b]
            assign[b] = -1
            load[r] -= weights[b]
            for nbr, w in nbrs[b]:
                aff[nbr, r] -= w
            return r

        # part 0 takes the seed, the rest seed by least affinity to what is seeded
        place(int(seed) if seed is not None else order[0], 0)
        for r in range(1, k):
            pool = [b for b in order if assign[b] == -1]
            pick = min(pool, key=lambda b: (aff[b, :r].sum(), -weights[b], b))
            place(pick, r)

        scale = max(edges.values(), default=0) or 1.0
        for b in order:
            if assign[b] != -1:
                continue
            # the part we talk to most, discounted by how loaded it already is
            score = aff[b] / scale - load / target
            score[load + weights[b] > cap] = -np.inf
            r = int(np.argmax(score))
            if np.isinf(score[r]):
                r = int(np.argmin(load))
            place(b, r)

        # moves alone cannot fix a balanced but wrong split, so swap as well
        for _ in range(self.npasses):
            moved = 0
            for b in range(nblks):
                r = assign[b]
                best, bestgain = r, 0.0
                for cand in set(int(assign[n]) for n, _ in nbrs[b]):
                    if cand == r or load[cand] + weights[b] > cap:
                        continue
                    gain = aff[b, cand] - aff[b, r]
                    if gain > bestgain:
                        best, bestgain = cand, gain
                if best != r:
                    unplace(b)
                    place(b, best)
                    moved += 1

            for (a, b), w in sorted(edges.items()):
                ra, rb = int(assign[a]), int(assign[b])
                if ra == rb:
                    continue
                gaina = aff[a, rb] - aff[a, ra]
                gainb = aff[b, ra] - aff[b, rb]
                gain = gaina + gainb - 2.0 * w
                if gain <= 0.0:
                    continue
                newrb = load[rb] + weights[a] - weights[b]
                newra = load[ra] + weights[b] - weights[a]
                if newrb > cap or newra > cap:
                    continue
                unplace(a)
                unplace(b)
                place(a, rb)
                place(b, ra)
                moved += 1
            if moved == 0:
                break

        # tighten the load spread, but never at the cost of more traffic
        for _ in range(self.npasses):
            hi, lo = int(np.argmax(load)), int(np.argmin(load))
            spread = load[hi] - load[lo]
            if spread <= 0:
                break
            bestpair, bestspread = None, spread
            for a in np.where(assign == hi)[0]:
                for b in np.where(assign == lo)[0]:
                    d = weights[a] - weights[b]
                    if d <= 0:
                        continue
                    newspread = abs((load[hi] - d) - (load[lo] + d))
                    gaina = aff[a, lo] - aff[a, hi]
                    gainb = aff[b, hi] - aff[b, lo]
                    w = edges.get(tuple(sorted((int(a), int(b)))), 0.0)
                    dtraffic = gaina + gainb - 2.0 * w
                    if newspread < bestspread and dtraffic >= 0.0:
                        bestpair, bestspread = (int(a), int(b)), newspread
            if bestpair is None:
                break
            a, b = bestpair
            unplace(a)
            unplace(b)
            place(a, lo)
            place(b, hi)
        return assign
