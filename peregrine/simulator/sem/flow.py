import numpy as np

from peregrine.simulator.sem.eddies import EddyField
from peregrine.simulator.sem.models import Profile, Sigmas, Stats, database, yplus1


class Flow:
    name = None
    data = None
    # the tables run from the wall to the centreline and are mirrored about delta for the second wall
    mirrored = False
    inputs = dict(Uo=None, viscosity=None, utau=None, scale=1.0)
    required = ("Uo", "viscosity", "utau")
    # cells per axis of the grid eddy centres are drawn from
    ngrid = 200

    def __init__(self, **scalars):
        unknown = set(scalars) - set(self.inputs)
        if unknown:
            raise TypeError(
                f"{self.name} takes {sorted(self.inputs)}, not {sorted(unknown)}"
            )
        missing = [k for k in self.required if scalars.get(k) is None]
        if missing:
            raise TypeError(f"{self.name} needs {missing}")
        self.inputs = {**self.inputs, **scalars}
        self.update({})

    def update(self, changes):
        unknown = set(changes) - set(self.inputs)
        if unknown:
            raise TypeError(
                f"{self.name} takes {sorted(self.inputs)}, not {sorted(unknown)}"
            )
        self.inputs.update(changes)
        self.scalars = {**self.inputs, "delta": self.delta}
        self.profile = Profile.from_flow(self)
        self.sigmas = Sigmas.from_flow(self)
        self.stats = Stats.from_flow(self)

    def section(self, family):
        name = self.data or self.name
        entry = database(name)
        if family not in entry:
            msg = f"{name}.yaml has no {family}; it has {sorted(entry)}"
            raise KeyError(msg)
        return entry[family]

    @property
    def delta(self):
        raise NotImplementedError

    # The cross-section ------------------------------------------------------

    def wallcoord(self, ys, zs):
        raise NotImplementedError

    def localframe(self, ys, zs):
        raise NotImplementedError

    def scanrange(self):
        raise NotImplementedError

    def contains(self, ys, zs, tol=1.0e-4):
        raise NotImplementedError

    def dist(self, ys, zs):
        raise NotImplementedError

    def grid(self, margin):
        raise NotImplementedError

    def periods(self):
        return {}

    def has_eddies(self, ys, zs):
        return self.contains(ys, zs)

    def overlaps(self, locs, sigmas):
        sy = np.max(sigmas[:, :, 1], axis=1)
        sz = np.max(sigmas[:, :, 2], axis=1)
        reach = np.maximum(sy, sz)
        return self.dist(locs[:, 1], locs[:, 2]) < reach

    def admit(self, ys, zs):
        if not np.all(self.contains(ys, zs)):
            raise ValueError("some of the points are outside the inlet")

    @staticmethod
    def signs(neddy, rng):
        draw = rng.uniform(low=-1, high=1, size=(neddy, 3))
        return np.where(draw <= 0.0, -1.0, 1.0)

    # row 0 the unit normal, row 1 the tangent x cross normal, in plane components
    @staticmethod
    def frames(ny, nz):
        frames = np.empty((ny.shape[0], 2, 2))
        frames[:, 0, 0] = ny
        frames[:, 0, 1] = nz
        frames[:, 1, 0] = -nz
        frames[:, 1, 1] = ny
        return frames

    def sigmas_frames(self, locs):
        d = self.wallcoord(locs[:, 1], locs[:, 2])
        return self.sigmas(d), self.localframe(locs[:, 1], locs[:, 2])

    def wrap(self, eddies):
        # eddies straddling the low end are copied to the high end and those straddling the high end dropped, so the density at the seam matches the interior
        for axis, length in self.periods().items():
            reach = eddies.maxsigma(axis)
            coord = eddies.locs[:, axis]
            keep = np.where(coord + reach < length)
            straddle = np.where((coord + reach > 0.0) & (coord - reach < 0.0))
            images = eddies.locs[straddle]
            images[:, axis] = images[:, axis] + length
            locs = np.concatenate((eddies.locs[keep], images))
            eps = np.concatenate((eddies.eps[keep], eddies.eps[straddle]))
            speeds = (
                None
                if eddies.speeds is None
                else np.concatenate((eddies.speeds[keep], eddies.speeds[straddle]))
            )
            scales, frames = self.sigmas_frames(locs)
            eddies = EddyField(locs, eps, scales, speeds, frames)
        return eddies

    @staticmethod
    def cells(ylo, yhi, zlo, zhi, n):
        dy = (yhi - ylo) / n
        dz = (zhi - zlo) / n
        yc = ylo + dy * (np.arange(n) + 0.5)
        zc = zlo + dz * (np.arange(n) + 0.5)
        yy, zz = np.meshgrid(yc, zc, indexing="ij")
        return yy.ravel(), zz.ravel(), dy * dz


class Rectangle(Flow):
    def __init__(self, ydim, zdim, periodicy=False, periodicz=False, **scalars):
        # the base builds its models in its constructor from delta, which the geometry gives
        self.ydim = ydim
        self.zdim = zdim
        self.periodicy = periodicy
        self.periodicz = periodicz
        super().__init__(**scalars)

    @property
    def eddytop(self):
        return self.ydim

    def wallcoord(self, ys, zs):
        return np.asarray(ys, dtype=np.float64)

    def localframe(self, ys, zs):
        return None

    def scanrange(self):
        yp1 = yplus1(self.scalars)
        if yp1 is None:
            return 0.0, self.ydim
        return yp1, self.ydim - yp1

    def contains(self, ys, zs, tol=1.0e-4):
        ys, zs = np.asarray(ys), np.asarray(zs)
        iny = (ys >= -tol * self.ydim) & (ys <= (1.0 + tol) * self.ydim)
        inz = (zs >= -tol * self.zdim) & (zs <= (1.0 + tol) * self.zdim)
        return iny & inz

    def has_eddies(self, ys, zs):
        return self.contains(ys, zs) & (np.asarray(ys) < self.eddytop)

    def dist(self, ys, zs):
        dy = np.maximum(np.maximum(-ys, ys - self.ydim), 0.0)
        dz = np.maximum(np.maximum(-zs, zs - self.zdim), 0.0)
        return np.sqrt(dy * dy + dz * dz)

    def overlaps(self, locs, sigmas):
        # exact for a rectangle: the footprint box overlaps the inlet
        sy = np.max(sigmas[:, :, 1], axis=1)
        sz = np.max(sigmas[:, :, 2], axis=1)
        iny = (locs[:, 1] + sy > 0.0) & (locs[:, 1] - sy < self.ydim)
        inz = (locs[:, 2] + sz > 0.0) & (locs[:, 2] - sz < self.zdim)
        return iny & inz

    def grid(self, margin):
        yhi, zhi = self.eddytop + margin, self.zdim + margin
        return self.cells(-margin, yhi, -margin, zhi, self.ngrid)

    def periods(self):
        lengths = {}
        if self.periodicy:
            lengths[1] = self.ydim
        if self.periodicz:
            lengths[2] = self.zdim
        return lengths


class Channel(Rectangle):
    name = "channel"
    mirrored = True

    def __init__(self, ydim, zdim, periodicz=False, **scalars):
        super().__init__(ydim, zdim, periodicz=periodicz, **scalars)

    @property
    def delta(self):
        return 0.5 * self.ydim


class BoundaryLayer(Rectangle):
    name = "boundary-layer"
    inputs = {**Flow.inputs, "delta": None}
    required = ("Uo", "viscosity", "utau", "delta")

    def __init__(self, ydim, zdim, periodicz=False, **scalars):
        super().__init__(ydim, zdim, periodicz=periodicz, **scalars)

    @property
    def delta(self):
        return self.inputs["delta"]

    @property
    def eddytop(self):
        return self.delta


class Isotropic(Rectangle):
    name = "isotropic"
    required = ("Uo",)

    def __init__(self, ydim, zdim, periodicy=True, periodicz=True, **scalars):
        super().__init__(ydim, zdim, periodicy, periodicz, **scalars)

    @property
    def delta(self):
        return None


def polar(ys, zs):
    ys, zs = np.asarray(ys, np.float64), np.asarray(zs, np.float64)
    return np.sqrt(ys * ys + zs * zs)


def radial(ys, zs, r, inward):
    safe = np.where(r > 0.0, r, 1.0)
    ny = np.where(r > 0.0, ys / safe, 1.0)
    nz = np.where(r > 0.0, zs / safe, 0.0)
    if inward:
        ny, nz = -ny, -nz
    return Flow.frames(ny, nz)


class Pipe(Flow):
    name = "pipe"
    data = "channel"

    def __init__(self, R, **scalars):
        # the base builds its models in its constructor from delta, which R gives
        self.R = R
        super().__init__(**scalars)

    @property
    def delta(self):
        return self.R

    def wallcoord(self, ys, zs):
        return self.R - polar(ys, zs)

    def localframe(self, ys, zs):
        ys, zs = np.asarray(ys, np.float64), np.asarray(zs, np.float64)
        return radial(ys, zs, polar(ys, zs), inward=True)

    def scanrange(self):
        return yplus1(self.scalars), self.R

    def contains(self, ys, zs, tol=1.0e-4):
        return polar(ys, zs) <= (1.0 + tol) * self.R

    def dist(self, ys, zs):
        return np.maximum(polar(ys, zs) - self.R, 0.0)

    def grid(self, margin):
        lo, hi = -self.R - margin, self.R + margin
        ys, zs, area = self.cells(lo, hi, lo, hi, self.ngrid)
        keep = self.dist(ys, zs) <= margin
        return ys[keep], zs[keep], area


class Annulus(Flow):
    name = "annulus"
    data = "channel"
    mirrored = True

    def __init__(self, Ri, Ro, **scalars):
        # the base builds its models in its constructor from delta, which the radii give
        self.Ri = Ri
        self.Ro = Ro
        super().__init__(**scalars)

    @property
    def delta(self):
        return 0.5 * (self.Ro - self.Ri)

    def wallcoord(self, ys, zs):
        return polar(ys, zs) - self.Ri

    def localframe(self, ys, zs):
        ys, zs = np.asarray(ys, np.float64), np.asarray(zs, np.float64)
        return radial(ys, zs, polar(ys, zs), inward=False)

    def scanrange(self):
        yp1, gap = yplus1(self.scalars), self.Ro - self.Ri
        return yp1, gap - yp1

    def contains(self, ys, zs, tol=1.0e-4):
        r = polar(ys, zs)
        return (r >= (1.0 - tol) * self.Ri) & (r <= (1.0 + tol) * self.Ro)

    def dist(self, ys, zs):
        r = polar(ys, zs)
        return np.maximum(np.maximum(self.Ri - r, r - self.Ro), 0.0)

    def grid(self, margin):
        lo, hi = -self.Ro - margin, self.Ro + margin
        ys, zs, area = self.cells(lo, hi, lo, hi, self.ngrid)
        keep = self.dist(ys, zs) <= margin
        return ys[keep], zs[keep], area
