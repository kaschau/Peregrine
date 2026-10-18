import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import yaml

from peregrine.misc import subclass_where

erf = np.vectorize(math.erf, otypes=[np.float64])

# a flow's scalars, what a model reads of it
scalars = ("Uo", "viscosity", "delta", "utau", "scale")


@cache
def database(name):
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(Path(__file__).parent / "database" / f"{name}.yaml") as f:
        return yaml.load(f, Loader=loader)


def yplus1(flow):
    utau = flow.get("utau")
    return None if utau is None else flow["viscosity"] / utau


def retau(flow):
    utau, delta = flow.get("utau"), flow.get("delta")
    if utau is None or delta is None:
        return None
    return utau * delta / flow["viscosity"]


def interp(x, knots, values):
    x = np.asarray(x, np.float64)
    scalar = x.ndim == 0
    xi = np.clip(np.atleast_1d(x), knots[0], knots[-1])
    idx = np.searchsorted(knots, xi, side="right") - 1
    idx = np.clip(idx, 0, knots.shape[0] - 2)
    x0, x1 = knots[idx], knots[idx + 1]
    dx = x1 - x0
    # a repeated knot, which a mirrored wall profile has at delta, is stepped over
    w = np.where(dx == 0.0, 0.0, (xi - x0) / np.where(dx == 0.0, 1.0, dx))
    w = w.reshape(w.shape + (1,) * (values.ndim - 1))
    yi = (1.0 - w) * values[idx] + w * values[idx + 1]
    return yi[0] if scalar else yi


class Model:
    # the data file's family this model answers for
    family = None
    # what a data section calls this law
    name = None

    def __init__(self, flow, section):
        self.flow = flow.scalars
        self.mirrored = flow.mirrored
        self.section = section
        self.knots, self.values = self.build()

    def build(self):
        return None, None

    def __call__(self, y):
        return interp(y, self.knots, self.values)

    @classmethod
    def from_flow(cls, flow):
        section = flow.section(cls.family)
        law = subclass_where(cls, name=section.get("law", "tabulated"))
        return law(flow, section)


class Tabulated:
    def table(self, names, odd=()):
        section, flow = self.section, self.flow
        delta, utau, nu = flow["delta"], flow["utau"], flow["viscosity"]
        y = np.asarray(section["y"], np.float64)
        columns = [np.asarray(section[n], np.float64) for n in names]
        channel = self.mirrored
        coordinate = section["coordinate"]
        if coordinate == "ydelta":
            if channel:
                y = np.concatenate((y, 2.0 - np.flip(y[:-1])))
            y = y * delta
        elif coordinate == "yplus":
            yp = y
            yd = yp / yp.max()
            # above 3 sqrt(Re_tau) the y+ scaling blends into delta scaling of the outer fraction, so a table at one Re_tau serves another
            overlap = np.where(yp > 3 * np.sqrt(utau * delta / nu))
            ramp = np.linspace(0, 1, overlap[0].shape[0])
            transition = np.exp(-np.exp(-7.5 * ramp + 2.0))
            y = yp * nu / utau
            wall = yp[overlap] * nu / utau
            outer = yd[overlap] * delta
            y[overlap] = wall * (1.0 - transition) + outer * transition
            if channel:
                y = np.concatenate((y, 2.0 * delta - np.flip(y[:-1])))
        else:
            raise ValueError(f"coordinate {coordinate!r}: ydelta or yplus")
        if channel:
            columns = [
                np.concatenate((c, (-1.0 if n in odd else 1.0) * np.flip(c[:-1])))
                for n, c in zip(names, columns)
            ]
        return y, columns


# The mean profile --------------------------------------------------------


class Profile(Model):
    family = "profile"


class TabulatedProfile(Tabulated, Profile):
    name = "tabulated"

    def build(self):
        y, (U,) = self.table(("U",))
        return y, U * self.flow["utau"]


class UniformProfile(Profile):
    name = "uniform"

    def __call__(self, y):
        return np.full(np.shape(y), self.flow["Uo"])


# the wake law follows Pope ch. 7 and "Revisiting the law of the wake in wall turbulence", J. Fluid Mech. 811 (2017)
class WallProfile(Profile):
    name = "wall"
    kappa = 0.4

    @staticmethod
    def check_retau(retau):
        # below Re_tau = 100 the log layer is empty and its knots would run backwards
        if 3.0 * np.sqrt(retau) <= 30.0:
            raise ValueError(
                f"Re_tau = {retau:.1f} has no log layer above y+ = 30 "
                f"(3 sqrt(Re_tau) = {3.0 * np.sqrt(retau):.1f}); this profile needs Re_tau > 100"
            )

    def build(self):
        flow, s = self.flow, self.section
        Uo, utau, nu, delta = flow["Uo"], flow["utau"], flow["viscosity"], flow["delta"]
        kappa = self.kappa
        retau = utau * delta / nu
        self.check_retau(retau)
        upinf = Uo / utau
        # the log law's intercept, set so that extended to delta it sits the wake deficit psi below Uo
        A = upinf - 1 / kappa * np.log(retau) - s["psi"]

        # viscous sublayer y+ = [0, 5]
        yplusVsl = np.linspace(0, 5, s["nsublayer"])
        uplusVsl = yplusVsl
        usVsl = uplusVsl * utau
        ysVsl = yplusVsl * nu / utau

        # log-law region y+ > 30, y+ <= 3 sqrt(Re_tau)
        yplusLlr = np.linspace(30, 3 * np.sqrt(retau), 100)
        uplusLlr = 1.0 / kappa * np.log(yplusLlr) + A
        usLlr = uplusLlr * utau
        ysLlr = yplusLlr * nu / utau

        # buffer layer y+ = [5, 30]: a quadratic Bezier, its control point where u+ = y+ meets the log law's tangent at y+ = 30
        corner = (uplusLlr[0] - 1 / kappa) / (1 - 1 / (kappa * 30))
        start = np.array([yplusVsl[-1], uplusVsl[-1]])
        control = np.array([corner, corner])
        end = np.array([yplusLlr[0], uplusLlr[0]])
        t = np.linspace(0.0, 1.0, 30)[:, None]
        first = (1.0 - t) * start + t * control
        second = (1.0 - t) * control + t * end
        bezier = (1.0 - t) * first + t * second
        yplusBufl, uplusBufl = bezier[:, 0], bezier[:, 1]
        usBufl = uplusBufl * utau
        ysBufl = yplusBufl * nu / utau

        # outer region, where the wake law blends the log law into Uo
        mu = s["wake-mu"] * delta
        sigma = s["wake-sigma"] * delta
        ysOr = np.linspace(ysLlr[-1], delta, 100)
        yplusOr = ysOr * utau / nu

        E = 0.5 * (1 + erf((ysOr - mu) / np.sqrt(2 * sigma**2)))
        # ease the start of the blend
        n = 4
        E[0:n] = E[0:n] * np.array([i / n for i in range(n)])
        E[0] = 0
        uplaw = 1.0 / kappa * np.log(yplusOr) + A
        uplusOr = upinf - (upinf - uplaw) * (1 - E)
        usOr = uplusOr * utau

        ys = np.concatenate((ysVsl, ysBufl, ysLlr, ysOr))
        us = np.concatenate((usVsl, usBufl, usLlr, usOr))
        if self.mirrored:
            ys = np.concatenate((ys, 2.0 * delta - np.flip(ys)))
            us = np.concatenate((us, np.flip(us)))
        return ys, us


# The eddy length scales --------------------------------------------------


@dataclass
class SigmaBounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    vmin: float
    vmax: float


# sigmas(y) is y.shape + (3, 3) indexed [component, direction], the directions streamwise, wall-normal, wall-tangent
class Sigmas(Model):
    family = "sigmas"

    @property
    def scale(self):
        return self.flow.get("scale", 1.0)

    def bounds(self, flow, npts=200):
        ylo, yhi = flow.scanrange()
        sigmas = self(np.linspace(ylo, yhi, npts))
        volumes = np.prod(sigmas, axis=2)
        return SigmaBounds(
            xmin=np.min(sigmas[:, :, 0]),
            xmax=np.max(sigmas[:, :, 0]),
            ymin=np.min(sigmas[:, :, 1]),
            ymax=np.max(sigmas[:, :, 1]),
            zmin=np.min(sigmas[:, :, 2]),
            zmax=np.max(sigmas[:, :, 2]),
            vmin=np.min(volumes),
            vmax=np.max(volumes),
        )

    @staticmethod
    def minvol(sigmas):
        return np.min(np.prod(sigmas, axis=2), axis=1)


class TabulatedSigmas(Tabulated, Sigmas):
    name = "tabulated"

    def build(self):
        flow = self.flow
        Uo, utau, delta = flow["Uo"], flow["utau"], flow["delta"]
        names = ("Tuu", "Tvv", "Tww", "Luu", "Lvv", "Lww")
        y, (Tuu, Tvv, Tww, Luu, Lvv, Lww) = self.table(names)
        sigmas = np.empty((y.shape[0], 3, 3))
        sigmas[:, 0, 0] = Tuu * delta / utau * Uo
        sigmas[:, 0, 1] = Luu * delta
        sigmas[:, 0, 2] = Luu * delta
        sigmas[:, 1, 0] = Tvv * delta / utau * Uo
        sigmas[:, 1, 1] = Lvv * delta
        sigmas[:, 1, 2] = Lvv * delta
        sigmas[:, 2, 0] = Tww * delta / utau * Uo
        sigmas[:, 2, 1] = Lww * delta
        sigmas[:, 2, 2] = Lww * delta
        return y, sigmas * self.scale


class UniformSigmas(Sigmas):
    name = "uniform"

    def __call__(self, y):
        shape = np.shape(np.asarray(y, np.float64))
        return np.full(shape + (3, 3), self.scale)

    def bounds(self, flow, npts=200):
        s = self.scale
        return SigmaBounds(s, s, s, s, s, s, s**3, s**3)


class LinearBLSigmas(Sigmas):
    name = "linear-bl"

    def build(self):
        yp1, delta = yplus1(self.flow), self.flow["delta"]
        y = np.array([yp1, delta])
        sigmas = np.empty((2, 3, 3))
        sigmas[0, :, :] = yp1
        sigmas[1, :, :] = delta
        return y, sigmas * self.scale


# The Reynolds stresses ----------------------------------------------------


class Stats(Model):
    family = "stats"


class TabulatedStats(Tabulated, Stats):
    name = "tabulated"

    def build(self):
        utau2 = self.flow["utau"] ** 2
        names = ("Ruu", "Rvv", "Rww", "Ruv", "Ruw", "Rvw")
        # v flips across a mirrored flow's centreline, so these change sign there
        odd = ("Ruv", "Rvw")
        y, (Ruu, Rvv, Rww, Ruv, Ruw, Rvw) = self.table(names, odd=odd)
        stats = np.empty((Ruu.shape[0], 3, 3))
        stats[:, 0, 0] = Ruu * utau2
        stats[:, 0, 1] = Ruv * utau2
        stats[:, 0, 2] = Ruw * utau2
        stats[:, 1, 0] = stats[:, 0, 1]
        stats[:, 1, 1] = Rvv * utau2
        stats[:, 1, 2] = Rvw * utau2
        stats[:, 2, 0] = stats[:, 0, 2]
        stats[:, 2, 1] = stats[:, 1, 2]
        stats[:, 2, 2] = Rww * utau2
        return y, stats


class IsotropicStats(Stats):
    name = "isotropic"

    def __call__(self, y):
        shape = np.shape(np.asarray(y, np.float64))
        return np.broadcast_to(np.eye(3), shape + (3, 3)).copy()
