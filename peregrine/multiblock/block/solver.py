from peregrine.multiblock.interface import SolverInterface
from peregrine.multiblock.block.restart import RestartBlock


class SolverBlock(RestartBlock):
    def tableval(self, name):
        return getattr(self, name, None)

    def _new_intface(self, nface):
        return SolverInterface(nface, self, self.backend)
