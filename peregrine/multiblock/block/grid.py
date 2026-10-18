import numpy as np

from peregrine.launch.ranges import CellCenterRange, FaceRange
from peregrine.multiblock.interface import GridInterface
from peregrine.multiblock.block.base import BaseBlock


def metric(kind, comps=(), **spec):
    def mark(maker):
        maker.arraykind = (kind, comps, spec)
        return maker

    return mark


class GridBlock(BaseBlock):
    # every low interface before every high one; the halo blend is order dependent
    _faceorder = (1, 3, 5, 2, 4, 6)

    # the coordinates; everything else is derived from them again
    carried = ("nodes",)

    def __init__(self, nblki, mb):
        super().__init__(nblki)
        self.mb = mb
        self.backend = mb.backend
        self.ng = mb.ng
        # every array is an attribute named for it, made once the extents are known
        for name in self.declared:
            setattr(self, name, None)

    def _new_intface(self, nface):
        return GridInterface(nface, self)

    ###########################################################################
    # The arrays a block has
    ###########################################################################
    @property
    def declared(self):
        return self.mb.columns

    @property
    def extents(self):
        return (self.ni, self.nj, self.nk)

    def set_extents(self, ni, nj, nk):
        super().set_extents(ni, nj, nk)
        for name in self.declared:
            setattr(self, name, self._alloc(name))

    def _alloc(self, name):
        return self.declared[name].alloc(self, self.backend, name)

    # a new array rather than a set in place: the old may have been released, and a table that held it must forget it
    def replace(self, name, values):
        array = self._alloc(name)
        array.set(values)
        setattr(self, name, array)

    @property
    def interior(self):
        ng = self.ng
        return np.s_[ng:-ng, ng:-ng, ng:-ng] if ng else np.s_[:, :, :]

    @property
    def nodeslab(self):
        i0, i1, j0, j1, k0, k1 = self.basebounds
        return np.s_[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1]

    @property
    def cellslab(self):
        i0, i1, j0, j1, k0, k1 = self.basebounds
        # nodes i0..i1 inclusive span cells i0..i1-1, so no + 1 here
        return np.s_[k0:k1, j0:j1, i0:i1]

    ###########################################################################
    # The metrics, from the nodes
    ###########################################################################
    @classmethod
    def metrics(cls):
        makers = ((n, getattr(cls, n)) for n in dir(cls) if n.startswith("_"))
        return [n[1:] for n, m in makers if hasattr(m, "arraykind")]

    def make_metrics(self, names):
        nodes = self.nodes.get()
        self.xcc.set(self._cellcenters(nodes))
        for name in names:
            getattr(self, name).set(getattr(self, f"_{name}")(nodes))

    @staticmethod
    def _corner(nodes, axes, *highs):
        s = [slice(None)] * 3
        for n, high in zip(axes, highs):
            s[n] = np.s_[1:] if high else np.s_[:-1]
        return nodes[tuple(s)]

    @staticmethod
    def _cellcenters(nodes):
        return 0.125 * (
            nodes[0:-1, 0:-1, 0:-1]
            + nodes[0:-1, 0:-1, 1::]
            + nodes[0:-1, 1::, 0:-1]
            + nodes[0:-1, 1::, 1::]
            + nodes[1::, 0:-1, 0:-1]
            + nodes[1::, 0:-1, 1::]
            + nodes[1::, 1::, 0:-1]
            + nodes[1::, 1::, 1::]
        )

    def _facecenters(self, nodes, a):
        inplane = [n for n in range(3) if n != a]
        highs = ((0, 0), (0, 1), (1, 0), (1, 1))
        c00, c01, c10, c11 = (self._corner(nodes, inplane, *h) for h in highs)
        return 0.25 * (c00 + c01 + c10 + c11)

    def _areavecs(self, nodes, a):
        # the cyclic axis pair, so the half cross of the diagonals points along the axis, out of the low face
        diagonal = [(a + 1) % 3, (a + 2) % 3]
        highs = ((0, 0), (0, 1), (1, 0), (1, 1))
        c00, c01, c10, c11 = (self._corner(nodes, diagonal, *h) for h in highs)
        return 0.5 * np.cross(c10 - c01, c11 - c00)

    @metric(FaceRange, 3, axis=0)
    def _ixc(self, nodes):
        return self._facecenters(nodes, 0)

    @metric(FaceRange, 3, axis=1)
    def _jxc(self, nodes):
        return self._facecenters(nodes, 1)

    @metric(FaceRange, 3, axis=2)
    def _kxc(self, nodes):
        return self._facecenters(nodes, 2)

    @metric(FaceRange, 3, axis=0)
    def _iS(self, nodes):
        return self._areavecs(nodes, 0)

    @metric(FaceRange, 3, axis=1)
    def _jS(self, nodes):
        return self._areavecs(nodes, 1)

    @metric(FaceRange, 3, axis=2)
    def _kS(self, nodes):
        return self._areavecs(nodes, 2)

    def _volumes(self, nodes):
        bodydiag = nodes[1::, 1::, 1::] - nodes[0:-1, 0:-1, 0:-1]
        S = [self._areavecs(nodes, a) for a in range(3)]
        high = S[0][1::] + S[1][:, 1::] + S[2][:, :, 1::]
        J = sum(bodydiag[..., n] * high[..., n] for n in range(3)) / 3.0
        # the halo's edge and corner cells, made of extrapolated nodes, may have none, and no kernel divides by theirs
        bad = int((J[self.interior] <= 0.0).sum())
        if bad:
            raise ValueError(f"block {self.nblki}: {bad} cells with no volume or a negative one")
        return J

    @metric(CellCenterRange)
    def _detJinv(self, nodes):
        # the kernels only ever divide by the volume; a halo cell of none gives an infinity nothing reads
        with np.errstate(divide="ignore"):
            return 1.0 / self._volumes(nodes)

    @metric(CellCenterRange, 3)
    def _dijk(self, nodes):
        lengths = []
        for a in range(3):
            centers = self._facecenters(nodes, a)
            far, near = [slice(None)] * 3, [slice(None)] * 3
            far[a], near[a] = np.s_[1:], np.s_[:-1]
            span = centers[tuple(far)] - centers[tuple(near)]
            length = np.sqrt((span**2).sum(axis=-1))
            # an axis one cell thick is not marched in: infinite, so no step limit binds on it
            thin = self.extents[a] == 2
            lengths.append(np.full_like(length, np.inf) if thin else length)
        return np.stack(lengths, axis=-1)

    @metric(CellCenterRange, (3, 3))
    def _Jinv(self, nodes):
        # the eight cell corners, by which end of i, j, k each takes
        c1, c2, c3, c4, c5, c6, c7, c8 = (
            self._corner(nodes, (0, 1, 2), *highs)
            for highs in (
                (0, 0, 0),
                (0, 1, 0),
                (1, 1, 0),
                (1, 0, 0),
                (0, 0, 1),
                (0, 1, 1),
                (1, 1, 1),
                (1, 0, 1),
            )
        )
        dxi = 0.25 * ((c4 - c1) + (c8 - c5) + (c3 - c2) + (c7 - c6))
        deta = 0.25 * ((c2 - c1) + (c3 - c4) + (c7 - c8) + (c6 - c5))
        dzeta = 0.25 * ((c5 - c1) + (c8 - c4) + (c6 - c2) + (c7 - c3))
        # the inverse jacobian is its adjugate, rows the cross products of the other two, over its determinant
        rows = np.cross(deta, dzeta), np.cross(dzeta, dxi), np.cross(dxi, deta)
        volumes = self._volumes(nodes)[..., None, None]
        # a halo edge or corner cell has no volume: an infinity nothing reads
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.stack(rows, axis=-2) / volumes

    def facenorms(self, axis):
        s = getattr(self, f"{axis}S").get()
        area = np.sqrt((s**2).sum(axis=-1))
        # a collapsed face has no area and no normal: nan, never a wrong unit vector
        with np.errstate(divide="ignore", invalid="ignore"):
            return area, [s[..., n] / area for n in range(3)]

    ###########################################################################
    # The node halo, from this block alone
    ###########################################################################
    def _plane(self, x, nface, index):
        return x[self.get_intface(nface).plane(index)]

    @staticmethod
    def _masks(shape, ng):
        r = np.pad(np.ones(shape[0], bool), ng)[:, None]
        c = np.pad(np.ones(shape[1], bool), ng)[None, :]
        return {"interface": r & c, "edge": r ^ c, "corner": ~(r | c)}

    # the mirror image through the boundary plane; a block no thicker than its halo has none, and steps from the two planes just inside
    @staticmethod
    def _layers(nface, n, ng, extent):
        if nface % 2:
            s0 = ng - n - 1
            if extent <= ng:
                return s0, s0 + 1, s0 + 2
            return s0, ng, ng + n + 1
        s0 = -ng + n
        if extent <= ng:
            return s0, s0 - 1, s0 - 2
        return s0, -ng - 1, -ng - n - 2

    @staticmethod
    def _blend(cur, extrap, hits):
        w = hits / (hits + 1.0)
        return np.where(hits == 0.0, extrap, w * cur + (1.0 - w) * extrap)

    def pad_halo(self, name):
        a = getattr(self, name).get()
        ng = self.ng
        for axis in range(3):
            planes = np.moveaxis(a, axis, 0)
            planes[:ng] = planes[ng]
            planes[len(planes) - ng :] = planes[len(planes) - ng - 1]
        getattr(self, name).set(a)

    def gen_halo(self):
        ng = self.ng
        extents = (self.ni, self.nj, self.nk)
        planes = ((self.nj, self.nk), (self.ni, self.nk), (self.ni, self.nj))
        masks = [self._masks(plane, ng) for plane in planes]
        x = self.nodes.get()
        # the halo is built from nothing, so it starts as nothing
        for nface, extent in zip((1, 3, 5), x.shape):
            self._plane(x, nface, np.s_[0:ng])[:] = 0.0
            self._plane(x, nface, np.s_[extent - ng :])[:] = 0.0
        # the interfaces, then the edges between them, then the corners: each pass reads what the one before it wrote
        for name in ("interface", "edge", "corner"):
            # a node is reached once, not once per coordinate
            hits = np.zeros(x.shape[:3])
            for nface in self._faceorder:
                axis = self.get_intface(nface).axis
                mask, extent = masks[axis][name], extents[axis]
                for n in range(ng):
                    s0, s1, s2 = self._layers(nface, n, ng, extent)
                    cur = self._plane(x, nface, s0)
                    counted = self._plane(hits, nface, s0)
                    near = self._plane(x, nface, s1)
                    far = self._plane(x, nface, s2)
                    extrap = 2.0 * near[mask] - far[mask]
                    seen = counted[mask][:, None]
                    cur[mask] = self._blend(cur[mask], extrap, seen)
                    counted[mask] += 1.0
        self.nodes.set(x)

    def move_periodics(self):
        periodic = [f for f in self.intfaces if f.rot is not None]
        if not periodic:
            return
        nodes = self.nodes.get()
        for interface in periodic:
            R, t = interface.rot, interface.xlate
            h = interface.halo(nodes)
            h[...] = (h.reshape(-1, 3) @ R.T + t).reshape(h.shape)
        self.nodes.set(nodes)
