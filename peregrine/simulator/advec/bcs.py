from peregrine.simulator.bcs import BaseBc


class BaseAdvecBc(BaseBc):
    def preadvflux(self, Q):
        arguments = dict(Q=Q, state="state", Qhalo=Q, statehalo="state")
        return self._kernels("preadvflux", **arguments)

    def postflux(self, Q):
        return []
