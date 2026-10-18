from peregrine.partition.merge import MergeMixin


class Conditioner(MergeMixin):
    def condition(self, mb):
        print("Conditioning the grid...")
        before = len(mb.blocks)
        removed = self.merge(mb)
        self.orient(mb)
        off = self.snap_nodes(mb)
        print(
            f"  merged away {removed} plane(s) of interfaces, {before} blocks -> {len(mb.blocks)}"
        )
        print("  every block re-indexed so its longest extent is i")
        print(f"  interfaces matched exactly, closing a gap of up to {off:.3e}")
