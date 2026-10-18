import numpy as np

from peregrine.simulator.mixture.constants import kB
from peregrine.simulator.mixture.polyfit import PolyFitMixin
from peregrine.simulator.mixture.transport.kinetic import KineticTheoryModel
from peregrine.simulator.mixture.diffusion.base import BaseSpeciesDiffusionModel


class BinaryModel(BaseSpeciesDiffusionModel, PolyFitMixin):
    name = "binary"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        self.needs += ("well", "diam", "dipole", "polarize")
        # the pair coefficients come off kinetic theory's collision parameters, which the dense models build on
        self.deps = (KineticTheoryModel,)

    def constants(self, species):
        """The pair fits by unordered pair, dijinv; the bath's species indices; and the kernel's
        layout of them: pairfit, a row per species of its partners' fits in walk order, and selffit, its own pair."""
        dijinv = self.padded(self.dijinv(species))
        bath = self.bath(species)
        ns = len(species)
        # each unordered pair's row of dijinv, triu_indices order, reached from either side
        slot = np.zeros((ns, ns), dtype=int)
        slot[np.triu_indices(ns)] = np.arange(ns * (ns + 1) // 2)
        slot = np.maximum(slot, slot.T)
        whole = len(bath) == ns
        # a species' partners in the kernel's walk order: every other species for the whole bath, the bath's for a partial one with its own slot left zero
        rows = [[m for m in range(ns) if m != n] if whole else list(bath) for n in range(ns)]
        pairfit = np.zeros((ns * len(rows[0]), dijinv.shape[1]))
        for n, r in enumerate(rows):
            for k, m in enumerate(r):
                if m != n:
                    pairfit[n * len(r) + k] = dijinv[slot[n, m]]
        return dict(super().constants(species), dijinv=dijinv, bath=bath, pairfit=pairfit, selffit=dijinv[np.diag(slot)])

    def bath(self, species):
        """The indices of the species a mixture coefficient sums over: the case's `bath`, or every species."""
        names = list(species)
        chosen = self.cfgsect["bath"]
        return np.array([names.index(n) for n in (names if chosen is None else chosen)], dtype=np.int32)

    def dijinv(self, species):
        m = KineticTheoryModel(self.cfgsect).ljparams(species)
        Ts, rmass, rwell, rdiam = m["Ts"], m["rmass"], m["rwell"], m["rdiam"]
        ns = len(rmass)
        k, j = np.triu_indices(ns)

        # (T, pair): reduced temperature, and the collision integrals there
        Tstar = np.outer(Ts, kB / rwell[k, j])
        delta = np.broadcast_to(m["rdelta"][k, j], Tstar.shape)
        omega11 = m["ci"].omega22(Tstar, delta) / m["ci"].astar(Tstar, delta)

        # at unit pressure; the kernel divides by the real one
        kT = kB * Ts[:, None]
        xsec = np.pi * rdiam[k, j] ** 2 * omega11
        diff = 3.0 / 16.0 * np.sqrt(2.0 * np.pi / rmass[k, j]) * kT**1.5 / xsec
        # the fit is of T^1.5 / D; the kernel restores the power
        diffinv = Ts[:, None] ** 1.5 / diff
        tol, deg = self.cfgsect["refit-tol"], self.cfgsect["refit-max-deg"]
        return [self.fit(np.log(Ts), Dinv, tol, deg)[0] for Dinv in diffinv.T]
