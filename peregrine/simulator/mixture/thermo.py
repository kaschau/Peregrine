import numpy as np

from peregrine.simulator.mixture.constants import Ru, Tref
from peregrine.misc import subclasses


class BaseThermoData:
    # the species key it reads
    name = None

    @classmethod
    def find(cls, sp):
        return next(f for f in subclasses(cls) if f.name in sp)

    @staticmethod
    def cp_r(sp):
        raise NotImplementedError

    @staticmethod
    def hsref_r(sp):
        raise NotImplementedError


class Janaf(BaseThermoData):
    name = "janaf"

    @staticmethod
    def cp_r(sp):
        j = sp["janaf"]
        return np.array(j["T"], float), np.array(j["cp"], float) / Ru

    @staticmethod
    def hsref_r(sp):
        j = sp["janaf"]
        return j["dHf298"] / Ru, j["s298"] / Ru


class Nasa7(BaseThermoData):
    name = "NASA7"

    @classmethod
    def cp_r(cls, sp):
        n7 = sp["NASA7"]
        T = np.linspace(n7["T-range"][0], n7["T-range"][-1], 200)
        return T, np.array([cls.cpR(n7, t) for t in T])

    @classmethod
    def hsref_r(cls, sp):
        n7 = sp["NASA7"]
        return cls.hRT(n7, Tref) * Tref, cls.sR(n7, Tref)

    @staticmethod
    def coeffs(n7, T):
        return n7["low"] if T <= n7["T-range"][1] else n7["high"]

    @classmethod
    def cpR(cls, n7, T):
        a = cls.coeffs(n7, T)
        return a[0] + T * (a[1] + T * (a[2] + T * (a[3] + T * a[4])))

    @classmethod
    def hRT(cls, n7, T):
        a = cls.coeffs(n7, T)
        poly = T * (a[1] / 2 + T * (a[2] / 3 + T * (a[3] / 4 + T * a[4] / 5)))
        return a[0] + poly + a[5] / T

    @classmethod
    def sR(cls, n7, T):
        a = cls.coeffs(n7, T)
        poly = T * (a[1] + T * (a[2] / 2 + T * (a[3] / 3 + T * a[4] / 4)))
        return a[0] * np.log(T) + poly + a[6]
