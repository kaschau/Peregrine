from peregrine.plugins.simulator import BaseSimulatorPlugin


class BaseSimulator:
    name = None
    bcscls = None
    solution = "Q"

    def __init__(self, cfg):
        self.cfg = cfg
        self.plugins = BaseSimulatorPlugin.claim(cfg["plugins"], self)

    def validate(self):
        pass

    def report(self):
        return f"  Simulator: {self.name}\n"

    @property
    def iovars(self):
        raise NotImplementedError

    def export(self, blk, names):
        raise NotImplementedError

    def ics(self):
        raise NotImplementedError

    def arrays(self):
        raise NotImplementedError

    def structs(self):
        return {}

    def tplargs(self):
        return {}

    def constants(self):
        return {}

    # each tag holds a maker of the kernel for the register it is given, not a kernel
    def decl_kerns(self, provider):
        self.provider = provider
        self.kernels = {}
        return self.kernels

    ###########################################################################
    # Slot graphs
    ###########################################################################
    def prestep_graph(self, g, Q, dQ):
        pass

    def poststep_graph(self, g):
        pass

    def initialize_graph(self, g):
        pass

    def prerhs_graph(self, g, Q, dQ=None):
        raise NotImplementedError

    def begin_dQ_graph(self, g, Q, dQ):
        pass

    def rhs_graph(self, g, Q, dQ, begun):
        raise NotImplementedError

    def postrhs_graph(self, g, Q):
        raise NotImplementedError
