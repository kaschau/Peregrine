from dataclasses import dataclass
from typing import Optional

import numpy as np

from peregrine.simulator.sem.models import Sigmas


@dataclass
class EddyField:
    locs: np.ndarray
    eps: np.ndarray
    sigmas: np.ndarray
    speeds: Optional[np.ndarray] = None
    frames: Optional[np.ndarray] = None

    def __len__(self):
        return self.locs.shape[0]

    @property
    def neddy(self):
        return self.locs.shape[0]

    def maxsigma(self, axis):
        return np.max(self.sigmas[:, :, axis], axis=1)

    def reach(self):
        y, z = self.maxsigma(1), self.maxsigma(2)
        if self.frames is None:
            return y, z
        # a turned footprint is bounded along either axis by the larger of the two scales
        both = np.maximum(y, z)
        return both, both

    def select(self, index):
        return EddyField(
            self.locs[index],
            self.eps[index],
            self.sigmas[index],
            None if self.speeds is None else self.speeds[index],
            None if self.frames is None else self.frames[index],
        )

    @classmethod
    def concatenate(cls, fields):
        fields = list(fields)

        def cat(name):
            parts = [getattr(f, name) for f in fields]
            missing = any(p is None for p in parts)
            return None if missing else np.concatenate(parts)

        return cls(*map(cat, ("locs", "eps", "sigmas", "speeds", "frames")))


# slabs are indexed by the time their eddies cross the inlet, so the slabs a window needs do not depend on the convection speeds
class SlabGrid:
    def __init__(self, Ts, tau0, seed):
        self.Ts = Ts
        self.tau0 = tau0
        self.seed = seed

    def index(self, tau):
        return int(np.floor((tau - self.tau0) / self.Ts))

    def window(self, j):
        lo = self.tau0 + j * self.Ts
        return lo, lo + self.Ts

    # seeded from (seed, j), so any rank at any time draws slab j bit for bit
    def rng(self, j):
        return np.random.default_rng([self.seed, j])


class SlabCache:
    def __init__(self, grid, build):
        self.grid = grid
        self.build = build
        self.slabs = {}
        self.digests = {}

    def needed(self, taulo, tauhi):
        jlo = max(0, self.grid.index(taulo))
        jhi = max(jlo, self.grid.index(tauhi))
        return range(jlo, jhi + 1)

    def eddies(self, indices):
        for j in indices:
            if j not in self.slabs:
                self.slabs[j], self.digests[j] = self.build(j)
        return EddyField.concatenate(self.slabs[j] for j in indices)

    def retire(self, j):
        for k in [k for k in list(self.slabs) if k < j]:
            del self.slabs[k]
            del self.digests[k]

    @property
    def live(self):
        return sorted(self.slabs)


class Convection:
    name = None

    def speed(self, Uo, profile, y):
        raise NotImplementedError


class UniformConvection(Convection):
    name = "uniform"

    def speed(self, Uo, profile, y):
        return np.full(np.shape(y), Uo)


class LocalConvection(Convection):
    name = "local"

    def speed(self, Uo, profile, y):
        return profile(y)


class Population:
    name = None

    def weights(self, sigmas, d):
        raise NotImplementedError

    def refvol(self, sigmas, bounds, d, pdf, area):
        raise NotImplementedError

    def arrivals(self, flow, bounds, ceddy, maxeddies, taulo, tauhi, speed, rng):
        margin = max(bounds.ymax, bounds.zmax)
        ys, zs, area = flow.grid(margin)
        d = flow.wallcoord(ys, zs)
        w = self.weights(flow.sigmas, d)
        pdf = w / (np.sum(w) * area)
        vref = self.refvol(flow.sigmas, bounds, d, pdf, area)
        U = speed(d)
        gridarea = ys.shape[0] * area
        swept = gridarea * (tauhi - taulo) * np.sum(pdf * U * area)
        neddy = int(ceddy * swept / vref)
        if neddy > maxeddies:
            raise ValueError(
                f"{self.name} population asks for {neddy:.3g} eddies "
                f"(c-eddy * swept volume / veddy = {ceddy} * {swept:.3g} / {vref:.3g}), "
                f"above max-eddies = {maxeddies:.3g}; reduce c-eddy, use the PDF "
                "method, or raise max-eddies"
            )
        # cells weighted by the convection speed too, so the density along x is the resting one
        pcell = pdf * U
        pcell = pcell / pcell.sum()
        cell = rng.choice(np.arange(ys.shape[0]), p=pcell, size=neddy)
        half = 0.5 * np.sqrt(area)
        ye = ys[cell] + rng.uniform(low=-half, high=half, size=neddy)
        ze = zs[cell] + rng.uniform(low=-half, high=half, size=neddy)
        taus = rng.uniform(low=taulo, high=tauhi, size=neddy)
        speeds = speed(flow.wallcoord(ye, ze))
        locs = np.column_stack((speeds * taus, ye, ze))
        return locs, speeds


class RandomPopulation(Population):
    name = "random"

    def weights(self, sigmas, d):
        return np.ones(d.shape)

    def refvol(self, sigmas, bounds, d, pdf, area):
        return bounds.vmin


class PdfPopulation(Population):
    name = "pdf"

    def weights(self, sigmas, d):
        veddy = Sigmas.minvol(sigmas(d))
        vmin, vmax = veddy.min(), veddy.max()
        vratio = vmax / vmin
        # weights run from one at the largest eddy to vratio at the smallest
        vprime = -(veddy - vmin) + (vmax - vmin)
        return (vprime / vmax) * vratio + 1.0

    def refvol(self, sigmas, bounds, d, pdf, area):
        veddy = Sigmas.minvol(sigmas(d))
        return np.sum(pdf * veddy * area)
