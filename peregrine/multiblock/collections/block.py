from peregrine.multiblock.collections.base import BaseCollection


class BlockCollection(BaseCollection):
    def _entries(self):
        return self.blocks

    def launches(self, kernel, name, spec=None, less=None, behind=None):
        return [[(k, self.table, self._tiling(k, name, spec, less, behind)) for k in group] for group in kernel.groups]

    def _tiling(self, kernel, name, spec, less, behind):
        if less is None and behind is None:
            return self.table.tiling(kernel, name)
        policy, ncomps = kernel.policy, kernel.ncomps
        if behind is not None:
            word, colls = "landed by", behind
        else:
            word, colls = ("filled by" if any(c.boundary for c in less) else "less"), less
        title = f"{name} {word} {spec.name}"
        # keyed by the family, not the bank: every bank's message brings the same halos, so one cut tiling serves them all
        key = (type(policy), kernel.axis, ncomps, title)

        def rangefn(blk):
            at = [pair for c in colls for pair in c.interfaces(blk, spec)]
            rng = policy.blkrange(blk)
            return rng.behind(name, at) if behind is not None else rng.less(name, at)

        return self.table.subtiling(key, self.table.entries, rangefn, policy.tilekind, ncomps, name=title)
