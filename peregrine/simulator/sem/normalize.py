import numpy as np

from peregrine.simulator.sem.chunk import eval_spline, spline


def whiten(signal):
    signal = signal - np.mean(signal, axis=0)
    ntimes = signal.shape[0]
    cov = np.einsum("t...i,t...j->...ij", signal, signal) / ntimes
    # a dead component is decoupled: its row and column go to the identity, so the factor stays invertible
    dead = np.einsum("...ii->...i", cov) <= 0.0
    alive = (~dead).astype(np.float64)
    cov = cov * alive[..., :, None] * alive[..., None, :]
    cov = cov + np.einsum("...i,ij->...ij", dead.astype(np.float64), np.eye(3))
    # jitter against exactly collinear components
    trace = np.einsum("...ii->...", cov)
    cov = cov + 1.0e-14 * trace[..., None, None] / 3.0 * np.eye(3)
    # the Cholesky factor, Gram-Schmidt in the order u, v, w, is smooth in the covariance, so the field keeps its spatial coherence
    whitening = np.linalg.inv(np.linalg.cholesky(cov))
    return np.einsum("...ij,t...j->t...i", whitening, signal)


def impose(raw, Rij, interp=False, nsub=10, batch=1024):
    ntimes, npts, _ = raw.shape
    L = np.linalg.cholesky(Rij)
    prime = np.empty_like(raw)
    for s in range(0, npts, batch):
        e = min(s + batch, npts)
        signal = raw[:, s:e, :]
        # the statistics are imposed on the cubic between frames, which is what the kernel evaluates; the frames are then kept
        if interp:
            knots = np.arange(ntimes, dtype=np.float64)
            fine = np.linspace(0, ntimes - 1, (ntimes - 1) * (nsub + 1) + 1)
            signal = eval_spline(knots, spline(knots, signal), fine)
            fidx = np.arange(ntimes) * (nsub + 1)
        else:
            fidx = slice(None)
        signal = whiten(signal)
        prime[:, s:e, :] = np.einsum("pij,tpj->tpi", L[s:e], signal)[fidx]
    return prime


def to_plane(prime, frames):
    n, t = frames[:, 0], frames[:, 1]
    out = prime.copy()
    out[:, :, 1] = prime[:, :, 1] * n[:, 0] + prime[:, :, 2] * t[:, 0]
    out[:, :, 2] = prime[:, :, 1] * n[:, 1] + prime[:, :, 2] * t[:, 1]
    return out
