import numpy as np


def spline(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    n = x.shape[0]
    if n < 4:
        msg = f"a not-a-knot cubic spline needs at least 4 knots, not {n}"
        raise ValueError(msg)
    dx = np.diff(x)
    dxr = dx.reshape((n - 1,) + (1,) * (y.ndim - 1))
    slope = np.diff(y, axis=0) / dxr

    # the tridiagonal system for the knot slopes: second-derivative continuity in the interior rows, not-a-knot in the end rows
    lower, diag, upper = np.zeros(n), np.empty(n), np.zeros(n)
    b = np.empty_like(y)
    diag[1:-1] = 2.0 * (dx[:-1] + dx[1:])
    upper[1:-1] = dx[:-1]
    lower[1:-1] = dx[1:]
    b[1:-1] = 3.0 * (dxr[1:] * slope[:-1] + dxr[:-1] * slope[1:])
    d = x[2] - x[0]
    diag[0], upper[0] = dx[1], d
    b[0] = ((dxr[0] + 2.0 * d) * dxr[1] * slope[0] + dxr[0] ** 2 * slope[1]) / d
    d = x[-1] - x[-3]
    diag[-1], lower[-1] = dx[-2], d
    b[-1] = (dxr[-1] ** 2 * slope[-2] + (2.0 * d + dxr[-1]) * dxr[-2] * slope[-1]) / d

    # Thomas algorithm, every trailing axis at once
    cp, dp = np.empty(n), np.empty_like(b)
    cp[0], dp[0] = upper[0] / diag[0], b[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i] * cp[i - 1]
        cp[i] = upper[i] / m
        dp[i] = (b[i] - lower[i] * dp[i - 1]) / m
    s = np.empty_like(b)
    s[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        s[i] = dp[i] - cp[i] * s[i + 1]

    t = (s[:-1] + s[1:] - 2.0 * slope) / dxr
    c = np.empty((4, n - 1) + y.shape[1:])
    c[0] = t / dxr
    c[1] = (slope - s[:-1]) / dxr - t
    c[2] = s[:-1]
    c[3] = y[:-1]
    return c


def eval_spline(x, c, xq):
    xq = np.asarray(xq, np.float64)
    i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, x.shape[0] - 2)
    dt = (xq - x[i]).reshape(xq.shape + (1,) * (c.ndim - 2))
    return ((c[0, i] * dt + c[1, i]) * dt + c[2, i]) * dt + c[3, i]


class SignalChunk:
    def __init__(
        self,
        index,
        tbreaks,
        frames,
        c,
        extension=None,
        mean=None,
        flow=None,
        digests=(),
        diagnostics=None,
    ):
        self.index = index
        self.tbreaks = np.asarray(tbreaks, dtype=np.float64)
        self.frames = frames
        self.c = c
        self.extension = extension
        self.mean = mean
        self.flow = flow
        self.digests = tuple(digests)
        self.diagnostics = diagnostics

    @property
    def tstart(self):
        return self.tbreaks[0]

    @property
    def tend(self):
        return self.tbreaks[-1]

    def covers(self, t):
        tol = 1.0e-9 * (self.tend - self.tstart)
        return self.tstart - tol <= t <= self.tend + tol

    def locate(self, t, clamp=False):
        t = float(t)
        if not self.covers(t) and not clamp:
            raise ValueError(
                f"t={t} is outside chunk {self.index}, valid on [{self.tstart}, {self.tend}]; "
                "pass clamp=True to clamp"
            )
        t = min(max(t, self.tstart), self.tend)
        i = np.searchsorted(self.tbreaks, t, side="right") - 1
        i = int(np.clip(i, 0, self.tbreaks.shape[0] - 2))
        return i, t - self.tbreaks[i]

    def coeffs(self, t, clamp=False):
        i, _ = self.locate(t, clamp)
        return self.tbreaks[i], self.c[:, i]

    def evaluate(self, t, clamp=False):
        i, dt = self.locate(t, clamp)
        c = self.c[:, i]
        return ((c[0] * dt + c[1]) * dt + c[2]) * dt + c[3]
