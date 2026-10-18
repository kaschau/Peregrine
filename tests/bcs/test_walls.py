import itertools

import numpy as np
import pytest

from tests.bcs.bcblock import assert_consistent, case, spoil

# isothermal walls near the block's temperature, far colder and far hotter
TWALLS = (1000.0, 300.0, 3000.0)
# every kind: whether the flow sticks to it, moves with it, and whether it holds a temperature
WALLS = {
    "slp-adia-wall": dict(sticks=False, moving=False, isothermal=False),
    "slp-isot-wall": dict(sticks=False, moving=False, isothermal=True),
    "no-slp-adia-wall": dict(sticks=True, moving=False, isothermal=False),
    "mov-adia-wall": dict(sticks=True, moving=True, isothermal=False),
    "no-slp-isot-wall": dict(sticks=True, moving=False, isothermal=True),
    "mov-isot-wall": dict(sticks=True, moving=True, isothermal=True),
}


def cases():
    out = []
    for gas, kind in itertools.product(("air", "CH4_O2"), WALLS):
        for T in TWALLS if WALLS[kind]["isothermal"] else (None,):
            tag = f"{gas}-{kind}" + ("" if T is None else f"-{T:g}K")
            out.append(pytest.param(gas, kind, T, id=tag))
    return out


class Walled:
    def __init__(self, backend, gas, kind, Twall):
        entry = {} if Twall is None else {"T": Twall}
        self.mb = case(backend, kind, "kepaec", gas, simulator="navier-stokes", entry=entry)
        self.blk = self.mb.blocks[0]
        self.entry = self.mb.cfg["bcvals"]["outer"]
        self.rules = WALLS[kind]

    # halo layer l beside interior cell l, nearest the interface first
    def layers(self, interface, array):
        ng = self.blk.ng
        inner = np.s_[ng:-ng, ng:-ng]
        halos = [h[inner] for h in interface.halo(array)]
        planes = np.moveaxis(array, interface.axis, 0)[ng:-ng]
        images = planes if interface.is_low else planes[::-1]
        return list(zip(halos, [p[inner] for p in images]))

    def normal(self, interface):
        _, normals = self.blk.facenorms("ijk"[interface.axis])
        return np.stack([interface.faceplane(c) for c in normals], axis=-1)

    def check_preadvflux(self):
        # a spoil state proves the slot derives it from the solution again before the walls read it
        spoil(self.mb, self.blk)
        self.mb.postrhs()
        assert_consistent(self.mb, self.blk)
        Q, state = self.blk.Q.get(), self.blk.state.get()
        fields = list(self.mb.sim.statefields)
        T, p = fields.index("T"), fields.index("p")
        sticks, isothermal = self.rules["sticks"], self.rules["isothermal"]
        wall = np.array([self.entry[c] for c in "uvw"]) if self.rules["moving"] else 0.0
        for interface in self.blk.intfaces:
            n = self.normal(interface)
            for halo, image in self.layers(interface, Q):
                uh = halo[..., 1:4] / halo[..., :1]
                ui = image[..., 1:4] / image[..., :1]
                if sticks:
                    assert np.allclose(uh, 2.0 * wall - ui)
                else:
                    un = np.sum(ui * n, axis=-1, keepdims=True)
                    assert np.allclose(uh, ui - 2.0 * un * n)
                Yh, Yi = halo[..., 5:] / halo[..., :1], image[..., 5:] / image[..., :1]
                assert np.allclose(Yh, Yi)
                if not isothermal:
                    # the adiabatic reflection keeps the density and the internal energy
                    assert np.array_equal(halo[..., 0], image[..., 0])
                    eh = halo[..., 4] - 0.5 * halo[..., 0] * np.sum(uh * uh, axis=-1)
                    ei = image[..., 4] - 0.5 * image[..., 0] * np.sum(ui * ui, axis=-1)
                    assert np.allclose(eh, ei)
            for halo, image in self.layers(interface, state):
                assert np.allclose(halo[..., p], image[..., p])
                if isothermal:
                    want = np.full_like(image[..., T], self.entry["T"])
                else:
                    want = image[..., T]
                assert np.allclose(halo[..., T], want)

    # a wall does nothing at the gradient hooks: the flux through its faces is its own
    def check_gradient_hooks(self):
        before = self.blk.Q.get().copy()
        self.mb.apply_bcs("pregrads")
        assert np.array_equal(self.blk.Q.get(), before)
        rng = np.random.default_rng(11)
        grads = rng.uniform(-1.0, 1.0, self.blk.grads.get().shape)
        self.blk.grads.set(grads)
        self.mb.apply_bcs("postgrads")
        assert np.array_equal(self.blk.grads.get(), grads)


@pytest.mark.parametrize("gas,kind,Twall", cases())
def test_wall(backend, gas, kind, Twall):
    walled = Walled(backend, gas, kind, Twall)
    walled.check_preadvflux()
    walled.check_gradient_hooks()
