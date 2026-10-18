import weakref

import numpy as np

from peregrine.config import ConfigError
from peregrine.launch.arrays import ArraySpec, Broadcast
from peregrine.launch.ranges import InterfacePlaneRange
from peregrine.simulator.advec.bcs import BaseAdvecBc
from peregrine.simulator.sem.inlet import SemInlet


class SemInletBc(BaseAdvecBc):
    @property
    def columns(self):
        coeffs = ArraySpec("semcoeffs", InterfacePlaneRange, (3, 4))
        tbreak = ArraySpec("tbreak", Broadcast, 1)
        return {**super().columns, "semcoeffs": coeffs, "tbreak": tbreak}

    def start(self, tme, interfaces):
        super().start(tme, interfaces)
        self.interfaces = interfaces
        self.origin, self.R = self._frame(self.values.given["frame"])
        self.y, self.z = [], []
        for interface in interfaces:
            blk = interface.blk
            # the face centres over the interface proper, in the signal's plane
            faces = getattr(blk, f"{'ijk'[interface.axis]}xc").get()
            centers = interface.faceplane(faces)
            rel = centers - self.origin
            self.y.append(rel @ self.R[:, 1])
            self.z.append(rel @ self.R[:, 2])
            self._check_inward(interface)
        # two host buffers per interface, used in turn, so the source of a non-waiting device copy outlives it
        self.hslabs, self.hbreaks = [], []
        for i in interfaces:
            pinned = i.backend.pinned
            self.hslabs.append([pinned(i.semcoeffs.shape) for _ in (0, 1)])
            self.hbreaks.append([pinned(i.tbreak.shape) for _ in (0, 1)])
        self.slot = 0
        self.signal = SemInletSignal(self, tme)

    def update(self):
        self.signal.update()

    def load_interval(self, tbreak, slab):
        self.slot ^= 1
        rows = zip(self.interfaces, self.hslabs, self.hbreaks, self.signal.slices)
        for interface, hosts, hbreaks, sl in rows:
            host, hbreak = hosts[self.slot], hbreaks[self.slot]
            a, b = interface.shape
            plane = np.moveaxis(slab[:, sl, :], 0, -1).reshape(a, b, 3, 4)
            # the map to grid components is linear, so it applies per order
            turned = np.einsum("mj,abjk->abmk", self.R, plane)
            host[...] = turned.reshape(host.shape)
            hbreak[...] = tbreak
            interface.tbreak.set(hbreak, wait=False)
            interface.semcoeffs.set(host, wait=False)

    def _frame(self, frame):
        origin = np.asarray(frame["origin"], np.float64)
        ey, ez = (np.asarray(frame[k], np.float64) for k in ("ey", "ez"))
        ey, ez = ey / np.linalg.norm(ey), ez / np.linalg.norm(ez)
        if abs(ey @ ez) > 1e-8:
            msg = "frame.ey and ez have to be orthogonal."
            raise ConfigError(f"bcvals.{self.name}", msg)
        return origin, np.column_stack([np.cross(ey, ez), ey, ez])

    def _check_inward(self, intface):
        _, n = intface.blk.facenorms("ijk"[intface.axis])
        # the stored normals point into the block on the low interfaces
        sign = 1.0 if intface.is_low else -1.0
        inward = sign * np.stack([intface.faceplane(c) for c in n], axis=-1)
        if np.mean(inward @ self.R[:, 0]) <= 0.0:
            msg = "frame.ey x ez points out of the block; swap ey and ez, or negate one."
            raise ConfigError(f"bcvals.{self.name}", msg)


# every rank generates its own piece; the field is a function of the seed and the time alone, so the pieces agree without a message
class SemInletSignal:
    def __init__(self, boundary, tme):
        self.boundary, self.tme = boundary, tme
        ys = np.concatenate([y.ravel() for y in boundary.y])
        zs = np.concatenate([z.ravel() for z in boundary.z])
        sizes = [y.size for y in boundary.y]
        starts = np.cumsum([0] + sizes[:-1])
        self.slices = [slice(lo, lo + n) for lo, n in zip(starts, sizes)]
        with np.errstate(under="ignore"):
            self.inlet = SemInlet(boundary.values.given["sem"], ys, zs)
        # the interval on the interfaces, by its start time; none yet
        self.loaded = None
        # the prefetch thread goes with the signal, or at exit
        weakref.finalize(self, self.inlet.close)

    def update(self):
        t = self.tme.get()
        with np.errstate(under="ignore"):
            tbreak, slab = self.inlet.coeffs(t)
        if tbreak != self.loaded:
            self.boundary.load_interval(tbreak, slab)
            self.loaded = tbreak
