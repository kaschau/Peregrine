from dataclasses import dataclass, field

import numpy as np


# a line with many empty times has its few nonzero frames inflated by the whitening
@dataclass
class Diagnostics:
    npts: int
    ntimes: int
    maxempty: int
    silent: list = field(default_factory=list)
    empty: list = field(default_factory=list)

    def summary(self):
        lines = []
        if self.silent:
            y, z = self.silent[0]
            lines.append(
                f"{len(self.silent)} of {self.npts} lines have no eddies at all "
                f"(first at y={y}, z={z})"
            )
        if self.empty:
            worst, y, z = max(self.empty)
            lines.append(
                f"{len(self.empty)} of {self.npts} lines have more than "
                f"{self.maxempty} of {self.ntimes} times with zero fluctuations "
                f"(worst {worst} at y={y}, z={z})"
            )
        return "\n".join(lines) if lines else None


def planeframes(n):
    frames = np.zeros((n, 2, 2))
    frames[:, 0, 0] = 1.0
    frames[:, 1, 1] = 1.0
    return frames


# every component's footprint is cut at the largest of the three scales in each direction, which is the reach that cull, overlaps and reach() assume
def blob(dists, sigmas):
    dists = np.asarray(dists, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    maxsigma = np.max(sigmas, axis=1)
    imax = np.argmax(sigmas, axis=1)
    big = np.cos(np.pi * dists / (2.0 * maxsigma))
    own = np.cos(np.pi * dists[:, np.newaxis, :] / (2.0 * sigmas))
    small = big[:, np.newaxis, :] * own
    ismax = np.arange(3)[np.newaxis, :, np.newaxis] == imax[:, np.newaxis, :]
    fx = np.where(ismax, big[:, np.newaxis, :], small)
    # the product over the three directions, left to right
    fxx = (fx[:, :, 0] * fx[:, :, 1]) * fx[:, :, 2]
    outside = np.any(np.abs(dists) > maxsigma, axis=1)
    fxx[outside] = 0.0
    return fxx


def march(ys, zs, eddies, taus, has_eddies, ptframes, maxempty=10):
    ntimes, npts = taus.shape[0], ys.shape[0]
    neddy = eddies.neddy
    eframes = planeframes(neddy) if eddies.frames is None else eddies.frames
    ptframes = planeframes(npts) if ptframes is None else ptframes

    reachy, reachz = eddies.reach()

    # sorted by y, so a point's window is found by binary search
    srt = np.argsort(eddies.locs[:, 1])
    locs, sigmas, eps = eddies.locs[srt], eddies.sigmas[srt], eddies.eps[srt]
    speeds, frames = np.asarray(eddies.speeds)[srt], eframes[srt]
    reachy, reachz = reachy[srt], reachz[srt]
    sx = np.max(sigmas[:, :, 0], axis=1)
    ye = locs[:, 1]
    reachmax = np.max(reachy) if ye.shape[0] > 0 else 0.0

    # the contiguous run of times each eddy is passing, found once
    with np.errstate(divide="ignore", invalid="ignore"):
        tlo = (locs[:, 0] - sx) / speeds
        thi = (locs[:, 0] + sx) / speeds
    degenerate = speeds <= 0.0
    if np.any(degenerate):
        tlo[degenerate] = -np.inf
        thi[degenerate] = np.inf
    jlos = np.searchsorted(taus, tlo, "right")
    jhis = np.searchsorted(taus, thi, "left")

    # zeros, not empty, so a line with no eddies returns zero fluctuations
    raw = np.zeros((ntimes, npts, 3))
    diagnostics = Diagnostics(npts, ntimes, maxempty)
    maxpairs = 5_000_000

    for i, (y, z) in enumerate(zip(ys, zs)):
        # the eddies whose footprint covers this point's line
        ilo = np.searchsorted(ye, y - reachmax, side="left")
        ihi = np.searchsorted(ye, y + reachmax, side="right")
        iny = np.abs(ye[ilo:ihi] - y) < reachy[ilo:ihi]
        inz = np.abs(locs[ilo:ihi, 2] - z) < reachz[ilo:ihi]
        inreach = np.where(iny & inz)
        line = ilo + inreach[0]
        if len(line) == 0:
            if has_eddies[i]:
                diagnostics.silent.append((y, z))
            continue
        xline, yline, zline = locs[line, 0], locs[line, 1], locs[line, 2]
        speed_line, frame_line = speeds[line], frames[line]
        nP, tP = ptframes[i, 0], ptframes[i, 1]

        # the (eddy, time) pairs of each eddy's run, evaluated at once and summed into their times, in chunks of pairs to cap memory
        jlo, jhi = jlos[line], jhis[line]
        pairs = jhi - jlo
        fxsum = np.zeros((ntimes, 3))
        nline = xline.shape[0]
        avg = max(1.0, int(np.sum(pairs)) / nline)
        step = max(1, int(maxpairs / avg))
        for s in range(0, nline, step):
            e = min(s + step, nline)
            counts = pairs[s:e]
            npairs = int(np.sum(counts))
            if npairs == 0:
                continue
            eidx = np.repeat(np.arange(s, e), counts)
            starts = np.cumsum(counts) - counts
            tidx = np.repeat(jlo[s:e] - starts, counts) + np.arange(npairs)

            # the displacement from point to eddy per pair; streamwise, an eddy is at x_e - U_e tau
            dists = np.empty((npairs, 3))
            dists[:, 0] = xline[eidx] - speed_line[eidx] * taus[tidx]
            dy, dz = yline[eidx] - y, zline[eidx] - z
            fr = frame_line[eidx]
            dists[:, 1] = dy * fr[:, 0, 0] + dz * fr[:, 0, 1]
            dists[:, 2] = dy * fr[:, 1, 0] + dz * fr[:, 1, 1]

            pidx = line[eidx]
            fx = blob(dists, sigmas[pidx])
            contrib = eps[pidx] * fx
            # the eddy's (v, w) into the point's frame
            nE, tE = fr[:, 0], fr[:, 1]
            v, w = contrib[:, 1].copy(), contrib[:, 2].copy()
            contrib[:, 1] = v * (nE @ nP) + w * (tE @ nP)
            contrib[:, 2] = v * (nE @ tP) + w * (tE @ tP)
            for comp in range(3):
                raw[:, i, comp] += np.bincount(
                    tidx, weights=contrib[:, comp], minlength=ntimes
                )
                fxsum[:, comp] += np.bincount(
                    tidx, weights=fx[:, comp], minlength=ntimes
                )

        # a time where the footprints sum to zero in some component counts as empty
        nempty = int(np.sum(np.any(fxsum == 0.0, axis=1)))
        if nempty > maxempty:
            diagnostics.empty.append((nempty, y, z))

    return raw, diagnostics
