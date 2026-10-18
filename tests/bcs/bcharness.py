import numpy as np

from tests.gases import primitives
from tests.bcs.bcblock import assert_consistent, case, spoil


class BaseBC:
    kind = None
    # the reconstruction the case runs, two halo layers under muscl
    recon = "muscl"

    # the slices of the primitive vector that share a rule
    _slice = {
        "p": np.s_[0],
        "u": np.s_[1],
        "v": np.s_[2],
        "w": np.s_[3],
        "velo": np.s_[1:4],
        "T": np.s_[4],
        "Y": np.s_[5:],
        "TY": np.s_[4:],
        "all": np.s_[:],
    }

    def __init__(self, backend, adv, gas):
        self.backend, self.adv, self.gas = backend, adv, gas
        self.mb = case(backend, self.kind, adv, gas, self.recon)
        self.blk = self.mb.blocks[0]
        self.ns = self.mb.sim.mixture.ns
        # the entry every interface carries, whose values are baked into the kernel
        self.entry = self.mb.cfg["bcvals"]["outer"]
        # the latest snapshot of the primitives a hook wrote
        self.host = {}

    def check(self):
        for interface in self.blk.intfaces:
            self.assert_euler_halo(interface)

    def assert_euler_halo(self, interface):
        raise NotImplementedError

    # the interfaces' halos are disjoint, so the hook runs on every one and the interface under test still sees only its own
    def run(self, hook):
        if hook == "preadvflux":
            # a spoil state proves postrhs derives it from the solution again before the boundaries read it
            spoil(self.mb, self.blk)
            self.mb.postrhs()
            assert_consistent(self.mb, self.blk)
        else:
            self.mb.apply_bcs(hook)
        self.host["q"] = primitives(self.mb, self.blk)

    def q(self, name):
        if name.startswith("Y") and name[1:].isdigit():
            return self.host["q"][..., 5 + int(name[1:])]
        return self.host["q"][..., self._slice[name]]

    # a low interface stores the inward normal, so the sign returned makes it point out of the block
    def normals(self, interface):
        _, normals = self.blk.facenorms("ijk"[interface.axis])
        n = tuple(interface.faceplane(c) for c in normals)
        return n, (-1.0 if interface.is_low else 1.0)

    # the halo's edges and corners are never written, so never checked
    def halo(self, interface, a):
        ng = self.blk.ng
        return [h[ng:-ng, ng:-ng] for h in interface.halo(a)]

    # nearest the interface first
    def interior(self, interface, a):
        ng = self.blk.ng
        planes = np.moveaxis(a, interface.axis, 0)[ng:-ng]
        inner = planes if interface.is_low else planes[::-1]
        return [p[ng:-ng, ng:-ng] for p in inner]

    def _bcVals(self, name):
        if name.startswith("Y") and name[1:].isdigit():
            species = self.mb.sim.mixture.spnames[int(name[1:])]
            return self.entry.get(species, 0.0)
        return self.entry[name]

    def each_species(self, interface, rule):
        for n in range(self.ns - 1):
            rule(interface, f"Y{n}")

    # every rule checks all halo layers; `where` restricts the check to a mask over the interface's plane
    def first_cell(self, interface, name, where=None):
        a = self.q(name)
        for h in self.halo(interface, a):
            self._close(h, self.interior(interface, a)[0], where)

    def extrapolate(self, interface, name, where=None, lo=None, hi=None):
        a = self.q(name)
        first = self.interior(interface, a)[0]
        layers = zip(self.halo(interface, a), self.interior(interface, a)[1:])
        for h, deeper in layers:
            want = 2.0 * first - deeper
            if lo is not None or hi is not None:
                want = np.clip(
                    want,
                    lo if lo is not None else -np.inf,
                    hi if hi is not None else np.inf,
                )
            self._close(h, want, where)

    def imposed(self, interface, name, where=None):
        a = self.q(name)
        for h in self.halo(interface, a):
            self._close(h, self._bcVals(name), where)

    def reflect(self, interface, where=None):
        n, _ = self.normals(interface)
        velo = [self.q(c) for c in "uvw"]
        for g in range(self.blk.ng):
            image = [self.interior(interface, c)[g] for c in velo]
            uDotn = sum(f * ni for f, ni in zip(image, n))
            for c, f, ni in zip(velo, image, n):
                self._close(self.halo(interface, c)[g], f - 2.0 * uDotn * ni, where)

    def alongNormal(self, interface):
        n, _ = self.normals(interface)
        velo = [self.q(c) for c in "uvw"]
        for g in range(self.blk.ng):
            Vb = sum(self.halo(interface, c)[g] * ni for c, ni in zip(velo, n))
            for c, ni in zip(velo, n):
                self._close(self.halo(interface, c)[g], Vb * ni)

    @staticmethod
    def _close(got, want, where=None):
        if where is not None:
            got, want = got[where], want[where]
        assert np.allclose(got, want)
