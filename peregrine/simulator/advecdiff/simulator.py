from peregrine.launch.ranges import CellCenterRange
from peregrine.simulator.advec import BaseAdvecSimulator


class BaseAdvecDiffSimulator(BaseAdvecSimulator):
    @property
    def ngrad(self):
        raise NotImplementedError

    def arrays(self):
        arrays = super().arrays()
        grads = dict(kind=CellCenterRange, comps=(self.ngrad, 3))
        # every row a gradient; the velocity rows start the tensor a periodic turns
        arrays["grads"] = dict(grads, exchanged=1, vecstart=0)
        return arrays

    ###########################################################################
    # Slot graphs
    ###########################################################################
    def postbc(self):
        return ("trans",)

    def rhs_graph(self, g, Q, dQ, begun):
        k = self.kernels
        g.add_bcs("pregrads", Q)
        g.add_launch("dqdxyz", k["dqdxyz"](Q), "interior", after=("pregrads",))
        g.add_bcs("postgrads", Q, after=("dqdxyz",))
        # sent after the boundaries set the halo's gradients: a switch at a boundary face reads this stage's, and the launches after the send keep the boundary cells
        g.add_send("grads-sent", "grads", after=("postgrads",))
        g.add_recv("grads-landed", "grads")
        # the advective flux's switch may read the gradients, so it is cut by their exchange like the diffusive flux that follows it
        g.add_launch("adv", k["advflux"](Q), "interior", after=("grads-sent",))
        g.add_launch("advl", k["advflux"](Q), "interior", after=("grads-landed",))
        g.add_launch("diff", k["diffflux"](Q), "interior", after=("adv", "grads-sent"))
        g.add_launch("diffl", k["diffflux"](Q), "interior", after=("advl", "grads-landed"))
        g.add_bcs("postflux", Q, after=("diff", "diffl"))
        g.add_launch("apply", k["applyflux"](dQ, append=begun), "interior", after=("postflux",))
