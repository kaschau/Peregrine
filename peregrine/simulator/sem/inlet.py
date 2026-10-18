import hashlib
import threading

import numpy as np

from peregrine.misc import subclass_where
from peregrine.simulator.sem.chunk import SignalChunk, spline
from peregrine.simulator.sem.eddies import Convection, EddyField, Population, SlabCache, SlabGrid
from peregrine.simulator.sem.flow import Flow
from peregrine.simulator.sem.march import march
from peregrine.simulator.sem.models import scalars
from peregrine.simulator.sem.normalize import impose, to_plane


class Prefetch(threading.Thread):
    def __init__(self, index, build, *args):
        super().__init__(name=f"sem chunk {index}", daemon=True)
        self.index = index
        self.build, self.args = build, args
        self.chunk = None
        self.error = None

    def run(self):
        try:
            self.chunk = self.build(*self.args)
        except BaseException as error:
            self.error = error

    def result(self):
        self.join()
        if self.error is not None:
            raise self.error
        return self.chunk


# the eddy field is a function of (seed, time) alone and a chunk's scalars are fixed when its generation starts, so any rank holding any points, and any restart, gives the same values at the same points; t0 must be the same everywhere
class SemInlet:
    def __init__(self, cfgsect, ys, zs):
        cfgsect = dict(cfgsect)
        # the section's hyphenated keys are the options' names, fused
        flow = dict(cfgsect.pop("flow"))
        flow = {k.replace("-", ""): v for k, v in flow.items()}
        self.flow = subclass_where(Flow, name=flow.pop("type"))(**flow)
        options = {k.replace("-", ""): v for k, v in cfgsect.items()}
        self._options(**options)

        ys, zs = np.array(ys, np.float64), np.array(zs, np.float64)
        if ys.shape != zs.shape:
            raise ValueError("ys and zs need the same shape")
        self.flow.admit(ys.ravel(), zs.ravel())
        self.shape = ys.shape
        self.ys, self.zs = ys.ravel(), zs.ravel()
        # the points' bounding box, which slabs are culled to
        self.patch = (ys.min(), ys.max(), zs.min(), zs.max())
        # the wall coordinate and frame of each point, fixed by the shape
        self.d = self.flow.wallcoord(self.ys, self.zs)
        self.ptframes = self.flow.localframe(self.ys, self.zs)
        self.has_eddies = self.flow.has_eddies(self.ys, self.zs)

        self.pending = None
        self._rebuild(self.flow.inputs)
        if self.ufloor is None:
            self.ufloor = 0.05 * self.flow.scalars["Uo"]
        if self.nframes <= 3:
            raise ValueError(
                "imposing statistics on the ensemble needs more than 3 frames"
            )
        if self.nblend >= self.nframes:
            raise ValueError("nblend must be smaller than nframes")

        # an eddy straddles the plane for at most taumax either side of its crossing: how far slabs reach past a chunk, and where the first starts
        self.taumax = self.bounds.xmax / self.minspeed()
        Ts = self.T if self.tslab is None else float(self.tslab)
        tau0 = -np.ceil(self.taumax / Ts) * Ts
        self.grid = SlabGrid(Ts, tau0, self.seed)
        self.cache = SlabCache(self.grid, self.slab)

        self.chunk = None
        self.future = None

    def _options(
        self,
        ensembletime,
        nframes,
        ceddy=1.0,
        population="pdf",
        convect="uniform",
        ufloor=None,
        interp=True,
        seed=10101,
        t0=0.0,
        tslab=None,
        nblend=8,
        prefetch=True,
        maxeddies=50_000_000,
    ):
        self.T = float(ensembletime)
        self.nframes = int(nframes)
        self.dtframe = self.T / (self.nframes - 1)
        self.ceddy = ceddy
        self.population = subclass_where(Population, name=population)()
        self.convection = subclass_where(Convection, name=convect)()
        # the lowest convection speed, so wall eddies under local convection retire
        self.ufloor = ufloor
        self.interp = interp
        self.seed = seed
        self.t0 = t0
        self.tslab = tslab
        self.nblend = int(nblend)
        self.prefetch = prefetch
        self.maxeddies = maxeddies

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _rebuild(self, inputs):
        self.flow.update(inputs)
        self.bounds = self.flow.sigmas.bounds(self.flow)

    # Speeds and slabs -------------------------------------------------------

    def speed(self, d):
        Uo = self.flow.scalars["Uo"]
        u = self.convection.speed(Uo, self.flow.profile, d)
        return np.maximum(u, self.ufloor)

    def minspeed(self):
        Uo = self.flow.scalars["Uo"]
        return Uo if self.convection.name == "uniform" else self.ufloor

    def slab(self, j):
        taulo, tauhi = self.grid.window(j)
        rng = self.grid.rng(j)
        locs, speeds = self.population.arrivals(
            self.flow,
            self.bounds,
            self.ceddy,
            self.maxeddies,
            taulo,
            tauhi,
            self.speed,
            rng,
        )
        sigmas, frames = self.flow.sigmas_frames(locs)
        keep = self.flow.overlaps(locs, sigmas)
        eps = self.flow.signs(int(np.count_nonzero(keep)), rng)
        frames = None if frames is None else frames[keep]
        eddies = EddyField(locs[keep], eps, sigmas[keep], speeds[keep], frames)
        eddies = self.flow.wrap(eddies)
        # digested before the cull, so it is the slab as drawn and agrees across ranks
        digest = hashlib.blake2b(digest_size=8)
        for array in (eddies.locs, eddies.eps, eddies.sigmas, eddies.speeds):
            digest.update(np.ascontiguousarray(array).tobytes())
        return self.cull(eddies), digest.hexdigest()

    def cull(self, eddies):
        reachy, reachz = eddies.reach()
        ye, ze = eddies.locs[:, 1], eddies.locs[:, 2]
        iny = (ye - reachy < self.patch[1]) & (ye + reachy > self.patch[0])
        inz = (ze - reachz < self.patch[3]) & (ze + reachz > self.patch[2])
        keep = iny & inz
        return eddies.select(keep)

    # Chunks -----------------------------------------------------------------

    def index(self, t):
        return max(0, int(np.floor((t - self.t0) / self.T)))

    def current(self, t):
        k = self.index(t)
        if self.chunk is None or self.chunk.index != k:
            ready = self.collect()
            if ready is not None and ready.index == k:
                self.chunk = ready
            else:
                previous = None
                for candidate in (self.chunk, ready):
                    if candidate is not None and candidate.index == k - 1:
                        previous = candidate
                self.chunk = self.build(k, previous, self.next_inputs())
        self.submit(k + 1)
        return self.chunk

    def coeffs(self, t, clamp=False):
        return self.current(t).coeffs(t, clamp)

    def next_inputs(self):
        if self.pending is not None:
            inputs, self.pending = self.pending, None
            return inputs
        return dict(self.flow.inputs)

    def submit(self, k):
        future = self.future
        if not self.prefetch or (future is not None and future.index == k):
            return
        if future is not None:
            self.collect()
        self.future = Prefetch(k, self.build, k, self.chunk, self.next_inputs())
        self.future.start()

    def collect(self):
        if self.future is None or self.future is threading.current_thread():
            return None
        future, self.future = self.future, None
        return future.result()

    def build(self, k, previous, inputs, standalone=False):
        self._rebuild(inputs)
        # larger length scales widen the reach of slabs either side of a chunk
        self.taumax = max(self.taumax, self.bounds.xmax / self.minspeed())

        nb = self.nblend
        # this chunk's frames plus the extension the next chunk blends against
        tstart = self.t0 + k * self.T
        tgen = tstart + np.arange(0, self.nframes + nb) * self.dtframe
        taus = tgen - self.t0
        taulo, tauhi = taus[0] - self.taumax, taus[-1] + self.taumax
        indices = self.cache.needed(taulo, tauhi)
        eddies = self.cache.eddies(indices)
        digests = tuple((j, self.cache.digests[j]) for j in indices)
        self.cache.retire(indices.start)

        raw, diagnostics = march(
            self.ys,
            self.zs,
            eddies,
            taus,
            self.has_eddies,
            self.ptframes,
        )
        prime = impose(raw, self.flow.stats(self.d), self.interp)
        frames = self.ptframes
        vel = to_plane(prime, frames) if frames is not None else prime.copy()
        mean = self.flow.profile(self.d)
        vel[:, :, 0] += mean

        knots, values, skip = tgen, vel, 0
        if k > 0 and nb > 0 and not standalone:
            if previous is None or previous.index != k - 1:
                # standalone: only its tail is needed, and blending it would recurse through every chunk before it
                previous = self.build(k - 1, None, inputs, standalone=True)
            # the head blended into the previous chunk's extension by a quintic smoothstep, whose first two derivatives vanish at both ends
            a = np.linspace(0.0, 1.0, nb + 1)
            a = a * a * a * (a * (6.0 * a - 15.0) + 10.0)
            a = a[:, None, None]
            vel[: nb + 1] = (1.0 - a) * previous.extension + a * vel[: nb + 1]
            # the previous chunk's last frames go into the fit too, so the boundary knot is interior to the spline
            knots = np.concatenate((previous.tbreaks[-(nb + 1) : -1], tgen))
            values = np.concatenate((previous.frames[-(nb + 1) : -1], vel))
            skip = nb

        c = spline(knots, values)
        return SignalChunk(
            k,
            tgen[: self.nframes],
            vel[: self.nframes],
            c[:, skip : skip + self.nframes - 1],
            extension=vel[self.nframes - 1 :],
            mean=mean,
            flow=dict(self.flow.scalars),
            digests=digests,
            diagnostics=diagnostics,
        )

    # The signal -------------------------------------------------------------

    def velocity(self, t, clamp=False):
        return self.current(t).evaluate(t, clamp).reshape(self.shape + (3,))

    def fluctuations(self, t, clamp=False):
        chunk = self.current(t)
        vel = chunk.evaluate(t, clamp).reshape(self.shape + (3,))
        mean = chunk.mean.reshape(self.shape)
        return vel[..., 0] - mean, vel[..., 1], vel[..., 2]

    # takes effect on the chunks generated after the call; every rank calls it at the same step, and a restart repeats the same calls before the same chunks
    def update(self, **changes):
        unknown = set(changes) - set(self.flow.inputs)
        if unknown:
            raise TypeError(
                f"update takes {sorted(self.flow.inputs)}, not {sorted(unknown)}"
            )
        base = self.flow.inputs if self.pending is None else self.pending
        self.pending = {**base, **changes}

    def minframes(self):
        # so that no eddy passes a point in fewer than four frames
        dtmax = self.bounds.xmin / self.flow.scalars["Uo"] / 4.0
        return int(np.ceil(self.T / dtmax)) + 1

    # of the current chunk only, so it agrees across ranks however far each one's prefetch has got
    def fingerprint(self):
        chunk = self.chunk
        if chunk is None:
            return None
        flow = tuple(chunk.flow.get(k) for k in scalars)
        return chunk.index, chunk.digests, flow

    def close(self):
        self.prefetch = False
        self.collect()
