from functools import cached_property

import numpy as np

from peregrine.codegen import Struct
from peregrine.config import ConfigError
from peregrine.launch.kernel import Kernel
from peregrine.launch.ranges import CellCenterRange
from peregrine.misc import merge_disjoint, subclass_where
from peregrine.simulator.advec import BaseAdvecSimulator
from peregrine.simulator.chemint import BaseChemistry, NoChemistry
from peregrine.simulator.iovars import IoVars
from peregrine.simulator.mixture import get_mixture
from peregrine.simulator.euler.bcs import BaseEulerBc

templates = f"{__package__}.kernels"


class EulerSimulator(BaseAdvecSimulator):
    name = "euler"
    bcscls = BaseEulerBc
    # the slots cflmax reduces, in order: the largest CFL number per unit step of each over the rank's cells
    cflspeeds = ("acoustic", "convective", "combined")
    # the package of the advective flux's formulations, and of this physics' flux-function files
    formulationtpls = f"{__package__}.kernels.flux"
    fluxtpls = f"{__package__}.kernels.flux"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.mixture = get_mixture(cfg)
        kind = cfg["chem-integration"]["kind"]
        cls = subclass_where(BaseChemistry, name=kind) if kind else NoChemistry
        self.chemint = cls(cfg["chem-integration"], self.mixture)

    def validate(self):
        super().validate()
        self.chemint.validate()

    def report(self):
        species = f"  Species: {self.mixture.spnames}\n"
        eos = f"  Equation of State: {self.cfg['mixture']['eos']}\n"
        return super().report() + species + eos

    ###########################################################################
    # The vectors: what a case starts from, what Q's rows are, what a result writes
    ###########################################################################
    @property
    def ne(self):
        return len(self.iovars.privars)

    # None marks a primitive a result must hold, 0.0 what a species a result lacks starts at
    @cached_property
    def iovars(self):
        species = self.mixture.spnames
        primitives = dict.fromkeys(("p", "u", "v", "w", "T"))
        primitives.update({s: 0.0 for s in species[:-1]})
        exported = ["rho", *primitives, species[-1]]
        vectors = {"Velocity": ("u", "v", "w")}
        return IoVars(primitives, exported, vectors, derive=self.export)

    @property
    def convars(self):
        rhoY = [f"rho{s}" for s in self.mixture.spnames[:-1]]
        return ["rho", "rhou", "rhov", "rhow", "rhoE"] + rhoY

    def export(self, blk, names):
        Q, state = blk.Q.get(), blk.state
        fields = self.statefields
        rho = Q[..., 0]
        # the halo's edges and corners are never written, so hold no density
        with np.errstate(divide="ignore", invalid="ignore"):
            rhoinv = 1.0 / rho
            species = self.mixture.spnames
            # p and T are pulled by field: a whole-state get copies every field of the wide state off the device
            values = {n: state.get(fields.index(n)) for n in ("p", "T")}
            values["rho"] = rho
            for n, name in enumerate("uvw"):
                values[name] = Q[..., 1 + n] * rhoinv
            for n, name in enumerate(species[:-1]):
                values[name] = Q[..., 5 + n] * rhoinv
            # the last species is what the others leave: all of it, for one
            others = sum(values[name] for name in species[:-1])
            values[species[-1]] = np.ones_like(rho) - others
        return {name: values[name] for name in names}

    def ics(self):
        ic = self.cfg["ics"]
        Y = ic["Y"]
        names = self.mixture.spnames
        unknown = sorted(set(Y) - set(names))
        if unknown:
            msg = f"names {unknown}, which are not species of this mixture: {names}."
            raise ConfigError("ics.Y", msg)
        total = sum(Y.values())
        if total > 1.0 + 1e-12:
            raise ConfigError("ics.Y", f"sums to {total}, more than one.")
        values = [ic[key] for key in ("p", "u", "v", "w", "T")]
        return values + [Y.get(name, 0.0) for name in names[:-1]]

    ###########################################################################
    # The state, the flux, and what the gas gives the templates
    ###########################################################################
    def arrays(self):
        arrays = super().arrays()
        arrays[self.solution]["vecstart"] = 1
        arrays["state"] = dict(kind=CellCenterRange, structs=self.statestructs)
        return merge_disjoint(arrays, self.chemint.arrays())

    @property
    def statestructs(self):
        return [self.mixture.structs()["EosState"]]

    # cached: the structs behind it are rebuilt from the mixture's constants at every ask
    @cached_property
    def statefields(self):
        return tuple(f for s in self.statestructs for f in s.fields)

    def structs(self):
        face = Struct("FaceState", ("u", "v", "w"), base="EosState")
        own = {**super().structs(), **self.mixture.structs(), "FaceState": face}
        return merge_disjoint(own, self.chemint.structs())

    def tplargs(self):
        args = dict(
            super().tplargs(),
            ns=self.mixture.ns,
            moments=self.mixture.eos.moments,
            eos=self.mixture.eos.template(),
            fluxtpls=self.fluxtpls,
        )
        return merge_disjoint(args, self.chemint.tplargs())

    def constants(self):
        return self.mixture.constants()

    @property
    def fluxbinds(self):
        return dict(state="state", dijk="dijk")

    def decl_kerns(self, provider):
        k = super().decl_kerns(provider)
        self.chemint.decl_kerns(provider)
        fromQ = f"{templates}.state.fromQ"
        k["fromQ"] = lambda Q: Kernel(provider, fromQ, Q=Q, state="state")
        fromprims = f"{templates}.state.fromprims"
        bound = dict(prims="q", state="state")
        k["fromprims"] = lambda Q: Kernel(provider, fromprims, Q=Q, **bound)
        cflmax = f"{templates}.cflmax"
        faces = dict(state="state", dijk="dijk", iS="iS", jS="jS", kS="kS")
        k["cflmax"] = lambda Q: Kernel(provider, cflmax, Q=Q, **faces)
        # the advective flux: the formulation's template, which includes the files the section names
        cfgsect = self.cfg["adv-flux"]
        advflux = f"{self.formulationtpls}.{cfgsect['formulation']}"
        k["advflux"] = lambda Q: self.facekernels(advflux, cfgsect, Q=Q, **self.fluxbinds)
        return k

    ###########################################################################
    # Slot graphs
    ###########################################################################
    def begin_dQ_graph(self, g, Q, dQ):
        self.chemint.begin_dQ_graph(g, Q, dQ)
