from peregrine.codegen import Struct
from peregrine.config import ConfigError
from peregrine.launch.kernel import Kernel
from peregrine.simulator.euler.simulator import EulerSimulator
from peregrine.simulator.mixture.eos import DoubleFluxRealGasModel

templates = f"{__package__}.kernels"


class EulerDoubleFluxSimulator(EulerSimulator):
    name = "euler-df"
    # the flux functions in the frozen planes, one file per flux function this extension supports
    fluxtpls = f"{__package__}.kernels.flux"

    def validate(self):
        super().validate()
        eos = self.mixture.eos
        if not isinstance(eos, DoubleFluxRealGasModel):
            msg = f"{eos.name!r}: {self.name} freezes the tangent plane of the real gas with one, real-gas-df"
            raise ConfigError("mixture.eos", msg)

    def report(self):
        return super().report() + "  Double Flux: on\n"

    # the energy row is written twice, in the left cell's plane at F[4] and in the right cell's at F[ne]
    @property
    def nflux(self):
        return self.ne + 1

    @property
    def statestructs(self):
        eos, ns = self.mixture.eos, self.mixture.ns
        plane = Struct("TangentPlane", eos.planefields(ns))
        return [*super().statestructs, plane]

    def decl_kerns(self, provider):
        fromplane = f"{templates}.fromplane"
        apply = f"{templates}.applyflux"
        freeze = f"{templates}.freeze"
        reads = dict(detJinv="detJinv", iF="iF", jF="jF", kF="kF")
        k = super().decl_kerns(provider)
        # the eos's own state kernel, kept for the settle before the first freeze when there is no plane yet
        k["fromeos"] = k["fromQ"]
        k["fromQ"] = lambda Q: Kernel(provider, fromplane, Q=Q, state="state")

        def applyflux(dQ, append=False):
            return Kernel(provider, apply, options=dict(append=append), **reads, dQ=dQ)

        k["applyflux"] = applyflux
        k["freeze"] = lambda Q: Kernel(provider, freeze, Q=Q, state="state")
        return k

    ###########################################################################
    # Slot graphs
    ###########################################################################
    def freeze_graph(self, g, Q, after=()):
        # halos too: a halo's state is its neighbor's since the last exchange, so its plane comes out the neighbor's without an exchange of planes
        g.add_launch("freeze", self.kernels["freeze"](Q), "all", after=after)

    def prestep_graph(self, g, Q, dQ):
        self.freeze_graph(g, Q)

    def initialize_graph(self, g):
        Q = self.solution
        readied = self.prerhs_graph(g, Q, derive=self.kernels["fromeos"], over="all")
        self.freeze_graph(g, Q, after=readied)
