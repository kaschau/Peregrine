from peregrine.launch.kernel import Kernel, KernelGroup
from peregrine.launch.ranges import CellCenterRange, FaceRange
from peregrine.simulator.base import BaseSimulator

templates = f"{__package__}.kernels"


class BaseAdvecSimulator(BaseSimulator):
    @property
    def ne(self):
        raise NotImplementedError

    @property
    def nflux(self):
        return self.ne

    def arrays(self):
        arrays = {}
        for axis, d in enumerate("ijk"):
            arrays[f"{d}F"] = dict(kind=FaceRange, comps=self.nflux, axis=axis)
        # exchanged: True trades the whole halo, an int that many planes (told apart by identity, since True == 1)
        soln = dict(kind=CellCenterRange, comps=self.ne, exchanged=True)
        arrays[self.solution] = soln
        return arrays

    def tplargs(self):
        return dict(super().tplargs(), ne=self.ne, nflux=self.nflux)

    def facekernels(self, template, cfgsect, **bindings):
        options = dict(cfgsect)
        bindings = dict(S="{a}S", F="{a}F", **bindings)
        kernels = []
        for d, a in enumerate("ijk"):
            faces = {k: v.format(a=a) for k, v in bindings.items()}
            kernels.append(Kernel(self.provider, template, axis=d, options=options, **faces))
        named = (f"{k} {v}" for k, v in options.items() if isinstance(v, str))
        return KernelGroup(kernels, name=" ".join(named))

    def decl_kerns(self, provider):
        k = super().decl_kerns(provider)
        tpl = f"{templates}.applyflux"
        reads = dict(detJinv="detJinv", iF="iF", jF="jF", kF="kF")

        def applyflux(dQ, append=False):
            return Kernel(provider, tpl, options=dict(append=append), **reads, dQ=dQ)

        k["applyflux"] = applyflux
        return k

    ###########################################################################
    # Slot graphs
    ###########################################################################

    def postbc(self):
        return ()

    def prerhs_graph(self, g, Q, dQ=None, derive=None, over="halos"):
        k = self.kernels
        derive = (k["fromQ"] if derive is None else derive)(Q)
        sent, landed = f"{Q}-sent", f"{Q}-landed"
        g.add_send(sent, Q)
        g.add_recv(landed, Q)
        g.add_launch("near", derive, over, after=(sent,))
        g.add_launch("far", derive, over, after=(landed,))
        # what begins dQ and the preadvflux boundaries fly alongside the exchange, so neither may read a halo it fills
        if dQ is not None:
            self.begin_dQ_graph(g, Q, dQ)
        g.add_bcs("preadvflux", Q)
        ends = ["near", "far", "preadvflux"]
        for tag in self.postbc():
            near, far = f"{tag}-near", f"{tag}-far"
            g.add_launch(near, k[tag](Q), "shallow", after=("near", "preadvflux", sent))
            g.add_launch(far, k[tag](Q), "shallow", after=("far", "preadvflux", landed))
            ends += [near, far]
        return tuple(ends)

    def begin_dQ_graph(self, g, Q, dQ):
        pass

    def rhs_graph(self, g, Q, dQ, begun):
        k = self.kernels
        g.add_launch("flux", k["advflux"](Q), "interior")
        g.add_bcs("postflux", Q, after=("flux",))
        g.add_launch("apply", k["applyflux"](dQ, append=begun), "interior", after=("postflux",))

    def postrhs_graph(self, g, Q):
        g.add_launch("fromq", self.kernels["fromQ"](Q), "interior")
