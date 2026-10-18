import numpy as np

from peregrine.partition.orient import OrientMixin


class MergeMixin(OrientMixin):
    def _plane(self, mb, nblki, nface):
        start = mb.get_blk(nblki).get_intface(nface)
        # a periodic interface is a boundary here
        if start.nbr is None or start.rot is not None:
            return None

        pairs, queue, seen = [], [(nblki, nface)], set()
        while queue:
            a, fa = queue.pop()
            blka = mb.get_blk(a)
            facea = blka.get_intface(fa)
            blkb = mb.get_blk(facea.nbr)
            key = frozenset({(a, fa), (blkb.nblki, facea.nbrnface)})
            if key in seen:
                continue
            seen.add(key)
            pairs.append((a, fa))

            for sidea in blka.intfaces:
                if sidea.axis == facea.axis:
                    continue
                fb = blka.nbrnfaceof(sidea.nface, across=fa)
                sideb = blkb.get_intface(fb)
                if sidea.nbr is None or sideb.nbr is None:
                    # the plane may end here, but at the same boundary on both blocks
                    if (sidea.nbr, sidea.bcname) == (sideb.nbr, sideb.bcname):
                        continue
                    return None
                if sidea.rot is not None or sideb.rot is not None:
                    return None
                # the block beside A must meet, across the plane, the block beside B
                blkc = mb.get_blk(sidea.nbr)
                fc = blka.nbrnfaceof(fa, across=sidea.nface)
                if blkc.get_intface(fc).nbr != sideb.nbr:
                    return None
                queue.append((blkc.nblki, fc))
        return pairs

    def planes(self, mb):
        planes, done = [], set()
        for blk, face in mb.intfaces():
            if (blk.nblki, face.nface) in done or face.nbr is None:
                continue
            pairs = self._plane(mb, blk.nblki, face.nface)
            if pairs is None:
                done.add((blk.nblki, face.nface))
                continue
            for a, fa in pairs:
                far = mb.get_blk(a).get_intface(fa)
                done.add((a, fa))
                done.add((far.nbr, far.nbrnface))
            planes.append(pairs)
        return planes

    def merge_plane(self, mb, pairs):
        merged = {}
        for a, fa in pairs:
            blka = mb.get_blk(a)
            facea = blka.get_intface(fa)
            blkb = mb.get_blk(facea.nbr)
            self.reorient(mb, blkb, *facea.align_nbr_by)

            axis = facea.axis
            lower, upper = (blkb, blka) if facea.is_low else (blka, blkb)
            unshared = [slice(None)] * 3
            unshared[axis] = slice(1, None)
            joined = {}
            for name in blka.carried:
                lo = getattr(lower, name).get()
                hi = getattr(upper, name).get()[tuple(unshared)]
                joined[name] = np.concatenate([lo, hi], axis=axis)
            dims = [blka.ni, blka.nj, blka.nk]
            dims[axis] += (blkb.ni, blkb.nj, blkb.nk)[axis] - 1
            blka.set_extents(*dims)
            for name, values in joined.items():
                getattr(blka, name).set(values)

            # B is in A's frame, so its far interface becomes A's
            blka.intfaces[fa - 1] = blkb.get_intface(fa)
            merged[blkb.nblki] = blka.nblki

        for blk, face in mb.intfaces():
            if face.nbr in merged:
                face.nbr = merged[face.nbr]
        self._compact(mb, set(merged))

    def _compact(self, mb, gone):
        keep = [blk for blk in mb.blocks if blk.nblki not in gone]
        renumber = {blk.nblki: n for n, blk in enumerate(keep)}
        for blk in keep:
            blk.nblki = renumber[blk.nblki]
            # a merged block is its own base again
            blk.origblk, blk.baseslice = blk.nblki, None
            for face in blk.intfaces:
                if face.nbr is not None:
                    face.nbr = renumber[face.nbr]
        mb.blocks[:] = keep
        mb.nblocks = len(keep)

    # a node on a block edge is shared by more than two blocks, so the copies are grouped and each group averaged once
    def snap_nodes(self, mb):
        # a compact id per node that lies on an interface, and where it lives
        ids, where = {}, []

        def idof(blk, flat):
            key = (blk.nblki, int(flat))
            if key not in ids:
                ids[key] = len(where)
                where.append(key)
            return ids[key]

        def flatids(blk):
            shape = blk.nodes.shape[:3]
            return np.arange(np.prod(shape)).reshape(shape)

        links, seen = [], set()
        for blk in mb.blocks:
            myflat = flatids(blk)
            for face in blk.intfaces:
                # a periodic pair's two sides are a transform apart, so their nodes are not averaged
                if face.nbr is None or face.rot is not None:
                    continue
                across = (face.nbr, face.nbrnface)
                key = frozenset({(blk.nblki, face.nface), across})
                if key in seen:
                    continue
                seen.add(key)

                other = mb.get_blk(face.nbr)
                theirface = other.get_intface(face.nbrnface)
                theirflat = flatids(other)
                mine = myflat[face.firstplane].ravel()
                theirs = face.align(theirflat[theirface.firstplane]).ravel()
                for a, b in zip(mine, theirs):
                    links.append((idof(blk, a), idof(other, b)))

        if not links:
            return 0.0

        # every copy of one node in one group
        n = len(where)
        rows, cols = np.array(links).T
        # scipy is a tool's dependency, not the solver's
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        _, group = connected_components(graph, directed=False)

        blocks = {blk.nblki: blk for blk in mb.blocks}
        held = np.array([w[0] for w in where])

        # each copy as (block, node index), on host copies of the nodes put back at the end
        hosts = {b: blocks[b].nodes.get() for b in set(held)}
        shapes = {b: h.shape[:3] for b, h in hosts.items()}
        sites = [(b, np.unravel_index(f, shapes[b])) for b, f in where]

        values = np.array([hosts[b][index] for b, index in sites])
        count = np.bincount(group)
        groupmean = [np.bincount(group, weights=v) / count for v in values.T]
        mean = np.stack(groupmean, axis=-1)[group]
        for m, (b, index) in enumerate(sites):
            hosts[b][index] = mean[m]
        for b, nodes in hosts.items():
            blocks[b].nodes.set(nodes)
        return float(np.abs(values - mean).max())

    def merge(self, mb):
        removed = 0
        while True:
            planes = self.planes(mb)
            if not planes:
                return removed
            self.merge_plane(mb, max(planes, key=len))
            removed += 1
