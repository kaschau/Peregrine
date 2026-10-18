import numpy as np

from peregrine.simulator.mixture.species import database


class CollisionIntegrals:
    def __init__(self):
        ci = database("integrals")
        self.delta = np.array(ci["delta"], dtype=np.float64)
        self._omega22 = self._splines(ci["tstar22"], ci["omega22"])
        self._astar = self._splines(ci["tstar"], ci["astar"])

    @staticmethod
    def _splines(tstar, table):
        # a table's row at T* = 0 is its limit, off the log axis
        keep = np.array(tstar, dtype=np.float64) > 0.0
        x = np.log(np.array(tstar, dtype=np.float64)[keep])
        y = np.array(table, dtype=np.float64)[keep]
        n = len(x)
        h = np.diff(x)
        # the tridiagonal system for the second derivatives, natural ends
        A = np.zeros((n, n))
        rhs = np.zeros((n, y.shape[1]))
        A[0, 0] = A[-1, -1] = 1.0
        for i in range(1, n - 1):
            h0, h1 = h[i - 1], h[i]
            A[i, i - 1], A[i, i], A[i, i + 1] = h0, 2.0 * (h0 + h1), h1
            rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h1 - (y[i] - y[i - 1]) / h0)
        return x, y, np.linalg.solve(A, rhs)

    @staticmethod
    def _spline(column, x, cols):
        knots, y, m = column
        i = np.clip(np.searchsorted(knots, x) - 1, 0, len(knots) - 2)
        h = knots[i + 1] - knots[i]
        a, b = (knots[i + 1] - x) / h, (x - knots[i]) / h
        y0, y1 = y[i, cols], y[i + 1, cols]
        m0, m1 = m[i, cols], m[i + 1, cols]
        cubic = (a**3 - a) * m0 + (b**3 - b) * m1
        return a * y0 + b * y1 + cubic * h**2 / 6.0

    def _at(self, column, tstar, delta):
        x = np.log(np.asarray(tstar, dtype=np.float64))
        delta = np.asarray(delta, dtype=np.float64)
        d = self.delta
        # the three columns about delta, kept inside the table
        j = np.clip(np.searchsorted(d, delta) - 1, 0, len(d) - 3)
        d0, d1, d2 = d[j], d[j + 1], d[j + 2]
        f0, f1, f2 = (self._spline(column, x, j + k) for k in range(3))
        t0 = f0 * (delta - d1) * (delta - d2) / ((d0 - d1) * (d0 - d2))
        t1 = f1 * (delta - d0) * (delta - d2) / ((d1 - d0) * (d1 - d2))
        t2 = f2 * (delta - d0) * (delta - d1) / ((d2 - d0) * (d2 - d1))
        return t0 + t1 + t2

    def omega22(self, tstar, delta):
        return self._at(self._omega22, tstar, delta)

    def astar(self, tstar, delta):
        return self._at(self._astar, tstar, delta)
