import itertools

import numpy as np
import pytest

import peregrine as pg

from tests.gases import configure, hcoeff, primitives
from tests.bcs.bcblock import case
from tests.bcs.bcharness import BaseBC


class Inlet(BaseBC):
    def assert_euler_halo(self, interface):
        self.run("preadvflux")
        self.state(interface)
        self.imposed(interface, "T")
        self.each_species(interface, self.imposed)


class SupersonicInlet(Inlet):
    kind = "sup-in-fa"

    def state(self, interface):
        self.imposed(interface, "p")
        for c in "uvw":
            self.imposed(interface, c)


class ConstantVelocitySubsonicInlet(Inlet):
    kind = "sub-in-fvt"

    def state(self, interface):
        self.extrapolate(interface, "p")
        for c in "uvw":
            self.imposed(interface, c)


class ConstantMassFluxSubsonicInlet(Inlet):
    kind = "sub-in-fmt"

    def state(self, interface):
        self.extrapolate(interface, "p")
        self._massFlux(interface)

    def _massFlux(self, interface):
        # only kepaec on a constant reconstruction carries exactly the average the kind wrote through the face
        mb = case(self.backend, self.kind, "kepaec", self.gas, "constant")
        blk = mb.blocks[0]
        stage = mb.integ.stageregs(mb.regs)[0]
        mb.rhs(Q=stage.Q, dQ=stage.dQ)
        d = "ijk"[interface.axis]
        own = blk.get_intface(interface.nface)
        F = own.faceplane(getattr(blk, f"{d}F").get())[..., 0]
        area, _ = blk.facenorms(d)
        S = own.faceplane(area)
        # the flux runs along the axis: into the block on a low interface
        mult = 1.0 if interface.is_low else -1.0
        target = mb.cfg["bcvals"]["outer"]["mflux"] * np.sum(S)
        assert abs(target - mult * np.sum(F)) / target < 1e-12


class StagnationSubsonicInlet(Inlet):
    kind = "sub-in-ftptt"

    def assert_euler_halo(self, interface):
        self.run("preadvflux")
        self.alongNormal(interface)
        self.each_species(interface, self.imposed)
        self.isentropic(interface)

    # without the reference enthalpy, which cancels in the difference it is used in
    def enthalpy(self, Y, T):
        mixture = self.mb.sim.mixture
        consts = mixture.constants()
        if mixture.eos.name == "cpg":
            return (Y @ consts["cp0"]) * T
        coeff = hcoeff(Y @ consts["cppolyR"])
        u = np.log(T)
        h = np.zeros_like(T)
        for m in reversed(range(coeff.shape[-1])):
            h = h * u + coeff[..., m]
        return h * T

    # the kernel's algebra transcribed: the halo from the entry's total enthalpy and the first cell's outgoing invariant on its own gamma and c
    def isentropic(self, interface):
        n, sign = self.normals(interface)
        inward = [-sign * c for c in n]
        fields = list(self.mb.sim.statefields)
        state = self.blk.state.get()
        gamma = self.interior(interface, state[..., fields.index("gamma")])[0]
        c = self.interior(interface, state[..., fields.index("c")])[0]
        T0 = self.interior(interface, self.q("T"))[0]
        u0 = [self.interior(interface, self.q(v))[0] for v in "uvw"]
        gm1 = gamma - 1.0
        Un = sum(u * ni for u, ni in zip(u0, inward))
        # the entry's total enthalpy in the invariant's convention: the cell's c^2 / (gamma - 1) plus the enthalpy from its temperature to the entry's, over the entry's composition
        carried = [self._bcVals(f"Y{k}") for k in range(self.ns - 1)]
        Y = np.array([*carried, 1.0 - sum(carried)])
        Ht = c * c / gm1 + self.enthalpy(Y, np.asarray(self.entry["Tt"], float)) - self.enthalpy(Y, T0)
        Jm = -Un + 2.0 * c / gm1
        aq = 1.0 + 2.0 / gm1
        bq = -2.0 * Jm
        cq = gm1 * (0.5 * Jm * Jm - Ht)
        t1 = -bq / (2.0 * aq)
        # a cell whose invariant cannot reach the entry's enthalpy takes the double root
        t2 = np.sqrt(np.maximum(bq * bq - 4.0 * aq * cq, 0.0)) / (2.0 * aq)
        cb = np.maximum(t1 + t2, t1 - t2)
        Vb = 2.0 * cb / gm1 - Jm
        isen = 1.0 + 0.5 * gm1 * (Vb / cb) ** 2
        pt, Tt = self.entry["pt"], self.entry["Tt"]
        for g in range(self.blk.ng):
            self._close(self.halo(interface, self.q("p"))[g], pt * isen ** (-gamma / gm1))
            self._close(self.halo(interface, self.q("T"))[g], Tt / isen)
            velo = [self.halo(interface, self.q(v))[g] for v in "uvw"]
            self._close(sum(u * ni for u, ni in zip(velo, inward)), Vb)


_inlets = (
    SupersonicInlet,
    ConstantVelocitySubsonicInlet,
    ConstantMassFluxSubsonicInlet,
    StagnationSubsonicInlet,
)


@pytest.mark.parametrize("adv,gas", list(itertools.product(("kepaec",), ("air", "CH4_O2"))))
@pytest.mark.parametrize("bc", _inlets, ids=lambda i: i.kind)
def test_inlet(backend, adv, gas, bc):
    bc(backend, adv, gas).check()


def test_a_stagnation_inlet_holds_its_isentropic_state(backend):
    pt, Tt, M = 101325.0, 300.0, 0.3
    config = pg.Config()
    configure(config, "air", "euler")
    config["adv-flux"]["recon"], config["adv-flux"]["limiter"] = "muscl", "koren"
    config["bcvals"]["inlet"] = {"kind": "sub-in-ftptt", "pt": pt, "Tt": Tt}
    config["bcvals"]["wall"] = {"kind": "slp-adia-wall"}
    mesh = pg.mesher.CubeMesher(
        mbdims=[1, 1, 1], blkdims=[6, 5, 4], lengths=[1, 1, 1], bcnames={1: "inlet"}, defaultbc="wall"
    )
    mb = pg.get_solver(config, mesh, backend(config))
    blk = mb.blocks[0]
    fields = list(mb.sim.statefields)
    cell = blk.state.get()[blk.interior][0, 0, 0]
    gamma = cell[fields.index("gamma")]
    R = cell[fields.index("p")] / (cell[fields.index("rho")] * cell[fields.index("T")])
    isen = 1.0 + 0.5 * (gamma - 1.0) * M * M
    T, p = Tt / isen, pt * isen ** (-gamma / (gamma - 1.0))
    q = primitives(mb, blk)
    q[..., 0], q[..., 1], q[..., 2:4], q[..., 4] = p, M * np.sqrt(gamma * R * T), 0.0, T
    mb.set_prims([q])
    mb.apply_bcs("preadvflux")
    q = primitives(mb, blk)
    inlet = blk.get_intface(1)
    ng = blk.ng
    inner = np.s_[ng:-ng, ng:-ng]
    first = q[ng][inner]
    for halo in inlet.halo(q):
        halo = halo[inner]
        assert np.allclose(halo[..., :5], first[..., :5], rtol=1e-12, atol=0.0)
        Th, ph, uh = halo[..., 4], halo[..., 0], halo[..., 1]
        Mh = uh / np.sqrt(gamma * R * Th)
        assert np.allclose(Th * (1.0 + 0.5 * (gamma - 1.0) * Mh**2), Tt, rtol=1e-12, atol=0.0)
        assert np.allclose(ph * (Tt / Th) ** (gamma / (gamma - 1.0)), pt, rtol=1e-12, atol=0.0)


def test_a_stagnation_inlet_holds_the_entrys_total_temperature_off_its_fixed_point(backend):
    pt, Tt, M = 101325.0, 300.0, 0.3
    config = pg.Config()
    configure(config, "air", "euler")
    config["bcvals"]["inlet"] = {"kind": "sub-in-ftptt", "pt": pt, "Tt": Tt}
    config["bcvals"]["wall"] = {"kind": "slp-adia-wall"}
    mesh = pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[6, 5, 4], lengths=[1, 1, 1], bcnames={1: "inlet"}, defaultbc="wall")
    mb = pg.get_solver(config, mesh, backend(config))
    blk = mb.blocks[0]
    fields = list(mb.sim.statefields)
    cell = blk.state.get()[blk.interior][0, 0, 0]
    gamma = cell[fields.index("gamma")]
    R = cell[fields.index("p")] / (cell[fields.index("rho")] * cell[fields.index("T")])
    isen = 1.0 + 0.5 * (gamma - 1.0) * M * M
    T, p = Tt / isen, pt * isen ** (-gamma / (gamma - 1.0))
    q = primitives(mb, blk)
    q[..., 0], q[..., 1], q[..., 2:4], q[..., 4] = p, M * np.sqrt(gamma * R * T), 0.0, 0.8 * T
    mb.set_prims([q])
    mb.apply_bcs("preadvflux")
    q = primitives(mb, blk)
    inlet = blk.get_intface(1)
    ng = blk.ng
    inner = np.s_[ng:-ng, ng:-ng]
    first = q[ng][inner]
    Mc = first[..., 1] / np.sqrt(gamma * R * first[..., 4])
    assert (first[..., 4] * (1.0 + 0.5 * (gamma - 1.0) * Mc**2) < 0.9 * Tt).all()
    for halo in inlet.halo(q):
        halo = halo[inner]
        Th, uh = halo[..., 4], halo[..., 1]
        Mh = uh / np.sqrt(gamma * R * Th)
        assert np.allclose(Th * (1.0 + 0.5 * (gamma - 1.0) * Mh**2), Tt, rtol=1e-12, atol=0.0)
