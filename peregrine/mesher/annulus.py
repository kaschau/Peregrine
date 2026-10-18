import numpy as np

from peregrine.mesher.base import BaseMesher


class AnnulusMesher(BaseMesher):
    name = "annulus"

    def __init__(
        self,
        p1=[0, 0, 0],
        p2=[1, 0, 0],
        p3=[0, 1, 0],
        sweep=45,
        thickness=0.1,
        periodic=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p1 = np.array(p1, dtype=np.float64)
        self.p2 = np.array(p2, dtype=np.float64)
        self.p3 = np.array(p3, dtype=np.float64)
        self.sweep = float(sweep)
        self.thickness = float(thickness)
        self.periodic = periodic

        length = np.linalg.norm(self.p2 - self.p1)
        rin = np.linalg.norm(self.p3 - self.p1)
        if length == 0.0:
            raise ValueError("p1 and p2 must differ, the line between them is the axis")
        if rin == 0.0:
            raise ValueError(
                "p3 must be off the axis, its distance is the inner radius"
            )

        self.axis = (self.p2 - self.p1) / length
        self.radial = (self.p3 - self.p1) / rin

        if abs(np.dot(self.axis, self.radial)) > 1e-7:
            raise ValueError("the line (p1,p3) is not orthogonal to (p1,p2)")
        # a negative sweep turns every cell inside out; flip p2 to go the other way round
        if not 0.0 < self.sweep <= 360.0:
            raise ValueError(
                f"sweep must be greater than 0 and no more than 360, not {sweep}"
            )
        # a rotational periodic is a pure rot about the origin, so the axis must pass through it
        if self.periodic and self.sweep != 360.0:
            offaxis = self.p1 - np.dot(self.p1, self.axis) * self.axis
            if np.linalg.norm(offaxis) > 1e-7:
                raise ValueError(
                    "a rotationally periodic annulus must be swept about an axis "
                    "through the origin"
                )

        # where each block of the lattice starts and ends: along the axis, out from it, and around it
        self.edges = [
            np.linspace(0.0, length, self.mbdims[0] + 1),
            np.linspace(rin, rin + self.thickness, self.mbdims[1] + 1),
            np.linspace(0.0, self.sweep, self.mbdims[2] + 1),
        ]

    @property
    def periodicaxes(self):
        return (False, False, self.periodic)

    def shape_blk(self, blk, i, j, k):
        blk.set_extents(*self.blkdims)

        axes = zip(self.edges, (i, j, k), self.blkdims)
        x, r, theta = (np.linspace(e[m], e[m + 1], n) for e, m, n in axes)
        radial = self._rotate(self.radial, np.radians(theta))

        axial = x[:, None, None, None] * self.axis
        outward = r[None, :, None, None] * radial[None, None, :, :]
        pts = self.p1 + axial + outward

        nodes = blk.nodes.get()
        nodes[blk.interior] = pts
        blk.nodes.set(nodes)

    def set_periodics(self, blk, i, j, k):
        if not self.periodic:
            return
        for edge, nface in ((0, 5), (self.mbdims[2] - 1, 6)):
            if k != edge:
                continue
            face = blk.get_intface(nface)
            if self.sweep == 360.0:
                face.set_interior(face.nbr, face.orient)
            else:
                # the low interface takes its halo from the far end of the sweep, so it is turned back; the high one the other way
                sweep = -self.sweep if face.is_low else self.sweep
                face.set_periodic(rot=face.rotabout(self.axis, sweep))

    def _rotate(self, p, theta):
        n = self.axis
        ct, st = np.cos(theta)[:, None], np.sin(theta)[:, None]
        # Rodrigues' formula, http://paulbourke.net/geometry/rotate/
        return ct * p + st * np.cross(n, p) + (1.0 - ct) * np.dot(n, p) * n
