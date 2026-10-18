from itertools import combinations

import numpy as np


class OrientMixin:
    def _longest(self, blk):
        perm = np.argsort([-blk.ni, -blk.nj, -blk.nk], kind="stable").tolist()
        # an odd permutation reverses k, so the block stays right handed
        swaps = sum(a > b for a, b in combinations(perm, 2))
        return perm, [False, False, swaps % 2 == 1]

    # new axis m runs along old axis perm[m], backwards when flips[m]
    def reorient(self, mb, blk, perm, flips):
        if perm == [0, 1, 2] and not any(flips):
            return

        face = blk.intfaces[0]
        invperm = {old: new for new, old in enumerate(perm)}

        # their strings name our axes, so each character becomes our new label
        for other in mb.blocks:
            if other is blk:
                continue
            for theirs in other.intfaces:
                if theirs.nbr != blk.nblki or theirs.orient is None:
                    continue
                codes = []
                for ours, flip in map(face.signedaxis, theirs.orient):
                    m = invperm[ours]
                    codes.append(face.orientcode(m, flip != flips[m]))
                theirs.orient = "".join(codes)

        # ours are indexed by our axes, so permuted not rewritten; a flip inverts
        for f in blk.intfaces:
            if f.orient is None:
                continue
            old, itself = f.orient, f.nbr == blk.nblki
            codes = []
            for m in range(3):
                axis, flip = face.signedaxis(old[perm[m]])
                flip = flip != flips[m]
                # one joined to its own block is theirs too, and the loop above skipped it
                if itself:
                    flip = flip != flips[invperm[axis]]
                    axis = invperm[axis]
                codes.append(face.orientcode(axis, flip))
            f.orient = "".join(codes)

        # an interface keeps its identity; its number follows its axis, a flip swaps ends
        reordered = [None] * 6
        for f in blk.intfaces:
            m = invperm[f.axis]
            low = f.is_low != flips[m]
            f.nface = 2 * m + (1 if low else 2)
            reordered[f.nface - 1] = f
        blk.intfaces = reordered

        # the carried arrays relabelled; set_extents makes new ones, which take them
        dims = (blk.ni, blk.nj, blk.nk)
        moved = {}
        for name in blk.carried:
            values = np.moveaxis(getattr(blk, name).get(), perm, (0, 1, 2))
            for m in range(3):
                if flips[m]:
                    values = np.flip(values, axis=m)
            moved[name] = values
        blk.set_extents(*(dims[perm[m]] for m in range(3)))
        for name, values in moved.items():
            getattr(blk, name).set(values)

    # the longest extent goes on i, the axis a kernel launch walks innermost
    def orient(self, mb):
        for blk in mb.blocks:
            self.reorient(mb, blk, *self._longest(blk))
