from peregrine.launch.kernel import Kernel
from peregrine.simulator.chemint.base import BaseReactingChemistry, templates


class CoupledChemistry(BaseReactingChemistry):
    name = "coupled"

    ###########################################################################
    # Slot graphs
    ###########################################################################
    def begin_dQ_graph(self, g, Q, dQ):
        source = Kernel(self.provider, f"{templates}.source", Q=Q, state="state", dQ=dQ)
        g.add_launch("source", source, "interior")
