import re
from concurrent.futures import ThreadPoolExecutor

from peregrine.misc import merge_disjoint
from peregrine.codegen.render import KernelRender, Unit


class KernelProvider:
    def __init__(self, backend, args):
        self.backend, self.args = backend, args
        # the case's halo depth once the solver has measured it: what a kernel may not read past
        self.ng = None

    # a no-op until the depth is known: the depth is measured from the stencils of the declared kernels
    def admit(self, kernel):
        if self.ng is not None and kernel.stencil > self.ng:
            what = f"{kernel} reads {kernel.stencil} halo layers"
            raise ValueError(f"{what}; the physics' depth is {self.ng}")

    def declare(self, kernel, columns=None):
        kernel.args = self.args
        if columns:
            arrays = merge_disjoint(self.args["arrays"], columns)
            kernel.args = {**self.args, "arrays": arrays}
        return KernelRender(kernel.args, kernel).run()

    def text(self, kernel):
        return Unit(kernel.args, kernel).text()

    def build(self, kernels):
        units = {}
        for k in kernels:
            units.setdefault(k.text, re.sub(r"\W+", "-", str(k)))
        with ThreadPoolExecutor() as pool:
            built = pool.map(self.backend.compiler.build, units.values(), units)
            paths = dict(zip(units, built))
        for k in kernels:
            k.bind(self.backend.function(paths[k.text], k))
