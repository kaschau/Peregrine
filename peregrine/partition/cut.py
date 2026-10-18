import numpy as np


class CutMixin:
    @staticmethod
    def _pair(pieces, pending):
        for lo_intface, hi_intface, nbr, flip in pending:
            low, high = pieces[nbr]
            if flip:
                low, high = high, low
            lo_intface.nbr, hi_intface.nbr = low, high

    def cut_blk(self, mb, nblki, cutaxis, cutidx):
        old = mb.get_blk(nblki)
        axis = "ijk".index(cutaxis)
        nnodes = (old.ni, old.nj, old.nk)[axis]

        # every piece needs a cell, so the last cut lands one short of the far end
        assert cutidx < nnodes - 1, (
            f"cannot cut block {nblki} along {cutaxis} at {cutidx}:"
            f" n{cutaxis} is {nnodes}, so the last cut is at {nnodes - 2}"
        )
        new = mb.add_blk()

        lonface = 2 * axis + 1
        hinface = 2 * axis + 2

        # the high half takes the old block's high interface before it becomes the cut
        oldhi = old.get_intface(hinface)
        newhi = new.get_intface(hinface)
        newhi.copy_from(oldhi)
        newhi.nbr = oldhi.nbr

        # the neighbor across that interface meets the new block
        if oldhi.nbr is not None:
            nbr = mb.get_blk(oldhi.nbr)
            nbr.get_intface(oldhi.nbrnface).nbr = new.nblki

        # the cut interfaces meet only each other; nothing one met before survives
        newlo = new.get_intface(lonface)
        oldhi.set_interior(new.nblki)
        newlo.set_interior(old.nblki)

        # the four interfaces along the cut split in two; only the neighbor is unknown
        pending = []
        for nface in (n for n in range(1, 7) if (n - 1) // 2 != axis):
            oldside = old.get_intface(nface)
            newside = new.get_intface(nface)
            newside.copy_from(oldside)
            if oldside.nbr is None:
                newside.nbr = None
                continue
            # our neighbor is cut too, so which halves meet waits for the path
            _, flip = oldside.signedaxis(oldside.orient[axis])
            pending.append((oldside, newside, oldside.nbr, flip))

        # cutidx is local, so it lands that far along whatever slab we already are
        base = list(old.basebounds)
        low, high = list(base), list(base)
        low[2 * axis + 1] = base[2 * axis] + cutidx
        high[2 * axis] = base[2 * axis] + cutidx
        new.origblk = old.origblk
        old.baseslice, new.baseslice = tuple(low), tuple(high)

        # split what the block carries before either half is resized; set_extents remakes its arrays
        halves = old.split_along(axis, cutidx)
        lodims = [old.ni, old.nj, old.nk]
        hidims = list(lodims)
        lodims[axis] = cutidx + 1
        hidims[axis] -= cutidx
        old.set_extents(*lodims)
        new.set_extents(*hidims)
        for var, (lo, hi) in halves.items():
            getattr(old, var).set(lo)
            getattr(new, var).set(hi)

        return pending

    @staticmethod
    def cuts(mb):
        table = np.empty((len(mb.blocks), 7), dtype=np.int32)
        for n, blk in enumerate(mb.blocks):
            table[n, 0] = blk.origblk
            table[n, 1:] = blk.basebounds
        return table

    def cutpath(self, mb, nblki, cutaxis):
        path = [[nblki, cutaxis, False]]
        queue = [[nblki, cutaxis, False]]

        while queue != []:
            b, ax, switch = queue.pop(0)
            blk = mb.get_blk(b)
            axis = "ijk".index(ax)

            # the cut runs out through the four interfaces it is parallel to
            for nface in (n for n in range(1, 7) if (n - 1) // 2 != axis):
                face = blk.get_intface(nface)
                nbr = face.nbr
                if nbr is None or nbr in [item[0] for item in path]:
                    continue
                theirs, flip = face.signedaxis(face.orient[axis])
                nbrswitch = switch != flip
                queue.append([nbr, "ijk"[theirs], nbrswitch])
                path.append([nbr, "ijk"[theirs], nbrswitch])

        return path

    def cut(self, mb, ops):
        print("Performing cut/s...")
        for nblki, cutaxis, ncuts in ops:
            blk = mb.get_blk(nblki)
            nx0 = getattr(blk, f"n{cutaxis}")
            print(f"  Cutting Block {nblki}'s {cutaxis} axis {ncuts} times.")

            for c in range(ncuts):
                path = self.cutpath(mb, nblki, cutaxis)
                nx = getattr(blk, f"n{cutaxis}")

                cutidx = int(nx0 * (ncuts - c) / (ncuts + 1))
                faridx = nx - cutidx - 1

                # which two blocks each block on the path became
                pieces, pending = {}, []
                for b, ax, switch in path:
                    assert getattr(mb.get_blk(b), f"n{ax}") == nx
                    idx = faridx if switch else cutidx
                    pending += self.cut_blk(mb, b, ax, idx)
                    pieces[b] = (b, mb.blocks[-1].nblki)

                self._pair(pieces, pending)
