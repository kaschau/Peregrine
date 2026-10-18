import itertools

import pytest

from tests.bcs.bcharness import BaseBC


class Exit(BaseBC):
    def _flowMasks(self, interface):
        n, sign = self.normals(interface)
        velo = [self.q(c) for c in "uvw"]
        first = [self.interior(interface, c)[0] for c in velo]
        uDotn = sum(f * ni for f, ni in zip(first, n)) * sign
        return uDotn > 0.0, uDotn <= 0.0

    def assert_euler_halo(self, interface):
        self.run("preadvflux")
        out, rev = self._flowMasks(interface)
        self.state(interface)
        for c in "uvw":
            self.extrapolate(interface, c, where=out)
        # reflected on the way in, so reverse flow cannot drag the halo inside
        self.reflect(interface, where=rev)


class ConstantPressureSubsonicExit(Exit):
    kind = "sub-out-fp"

    def state(self, interface):
        self.imposed(interface, "p")
        self.first_cell(interface, "T")
        if self.ns > 1:
            self.each_species(interface, self.first_cell)


class SupersonicExit(Exit):
    kind = "sup-out-fn"

    def state(self, interface):
        # p and T floor at a hundredth of the interior, so the halo keeps a density
        p = self.interior(interface, self.q("p"))[0]
        T = self.interior(interface, self.q("T"))[0]
        self.extrapolate(interface, "p", lo=0.01 * p, hi=p)
        self.extrapolate(interface, "T", lo=0.01 * T)
        for n in range(self.ns - 1):
            self.extrapolate(interface, f"Y{n}", lo=0.0, hi=1.0)


_exits = (ConstantPressureSubsonicExit, SupersonicExit)

pytestmark = pytest.mark.parametrize(
    "adv,gas",
    list(itertools.product(("kepaec",), ("air", "CH4_O2"))),
)


@pytest.mark.parametrize("bc", _exits, ids=lambda e: e.kind)
def test_exit(backend, adv, gas, bc):
    bc(backend, adv, gas).check()
