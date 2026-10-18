import numpy as np

from peregrine.launch.ranges import CellCenterRange, NodeRange
from peregrine.multiblock.base import BaseMultiBlock
from peregrine.multiblock.block import GridBlock


class Grid(BaseMultiBlock):
    def __init__(self, ng=0, backend=None):
        super().__init__(backend)
        self.ng = ng
        self.decl_array("nodes", NodeRange, comps=3, exchanged=True)
        self.decl_array("xcc", CellCenterRange, comps=3)
        # every metric a block can make, each declared as its maker says
        self.metrics = GridBlock.metrics()
        for name in self.metrics:
            kind, comps, spec = getattr(GridBlock, f"_{name}").arraykind
            self.decl_array(name, kind, comps, **spec)

    def _new_blk(self, nblki):
        return GridBlock(nblki, self)

    def mark_periodics(self, tol=1e-8):
        found = {}
        for blk, interface in self.intfaces():
            if interface.nbr is None or interface.rot is not None:
                continue
            other = self.get_blk(interface.nbr)
            mine = blk.nodes.get()[interface.firstplane].reshape(-1, 3)
            partner = other.get_intface(interface.nbrnface)
            plane = other.nodes.get()[partner.firstplane]
            theirs = interface.align(plane).reshape(-1, 3)

            moved = self._xform(theirs, mine, tol)
            if moved is None:
                continue
            rot, xlate = moved
            interface.set_periodic(rot=rot, xlate=xlate)
            found[interface.intkind] = found.get(interface.intkind, 0) + 1
        return found

    @staticmethod
    def _xform(theirs, mine, tol):
        delta = mine - theirs
        xlate = delta.mean(axis=0)
        if np.abs(delta).max() <= tol:
            return None
        if np.abs(delta - xlate).max() <= tol:
            return np.eye(3), xlate

        # a rotation about an axis through the origin: the orthogonal Procrustes solution
        u, _, vt = np.linalg.svd(mine.T @ theirs)
        rot = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
        off = np.abs(theirs @ rot.T - mine).max()
        if off > tol:
            raise ValueError(
                "an interface is neither joined nor a rigid transform apart:"
                f" the best rotation still leaves it {off:.3e} out"
            )
        return rot, np.zeros(3)

    def make_metrics(self):
        for blk in self.blocks:
            blk.make_metrics(self.metrics)

    def gen_halo(self):
        if self.ng == 0:
            raise ValueError("this grid has no halo: make it with Grid(ng=...)")
        for blk in self.blocks:
            blk.gen_halo()

    def move_periodics(self):
        for blk in self.blocks:
            blk.move_periodics()
