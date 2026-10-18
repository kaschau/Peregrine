import numpy as np


class PolyFitMixin:
    # coefficients come out ascending in the power, np.polyfit's reversed
    @staticmethod
    def fit(x, y, tol, maxdeg):
        # the tolerance is relative; of no weight, 1/y and 1/y^2, this reaches it at the lowest degree
        weights = 1.0 / y**2
        for degree in range(maxdeg + 1):
            c = np.polyfit(x, y, degree, w=weights)
            error = np.abs(np.polyval(c, x) / y - 1).max()
            if error <= tol:
                break
        return np.flip(c), error

    @staticmethod
    def evaluate(poly, x):
        return np.polyval(np.flip(poly), x)

    # Q with integral(P(u) e^u du) = e^u Q(u): for u = ln T, integral(P dT) = T Q
    @staticmethod
    def expintegral(P):
        # by parts: Q = P - P' + P'' - ...
        Q = np.zeros(len(P))
        D = np.flip(P)
        for k in range(len(P)):
            Q[: len(D)] += (-1) ** k * np.flip(D)
            D = np.polyder(D)
        return Q

    @staticmethod
    def integral(P):
        return np.flip(np.polyint(np.flip(P)))
