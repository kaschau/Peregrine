from peregrine.simulator.advec.bcs import BaseAdvecBc
from peregrine.simulator.euler.bcs import InletBc, IsoTWallBc
from peregrine.simulator.sem.boundary import SemInletBc


class GradientHooks:
    def pregrads(self, Q):
        return []

    def postgrads(self, Q):
        return self._kernels("postgrads", grads="grads", gradshalo="grads")


class WallHooks:
    # nothing reads a wall's halo gradients: postflux writes the flux through its faces from the interior cell
    def postgrads(self, Q):
        return []

    def postflux(self, Q):
        return self._kernels("postflux", faces=True, Q=Q, state="state", trstate="trstate", xcc="xcc", F="{a}F")


class BaseNavStokesBc(GradientHooks, BaseAdvecBc):
    pass


class WallBc(WallHooks, BaseNavStokesBc):
    pass


class AdiabaticSlipWall(WallBc):
    kind = "slp-adia-wall"


class IsoTSlipWall(IsoTWallBc, WallBc):
    kind = "slp-isot-wall"


class AdiabaticNoSlipWall(WallBc):
    kind = "no-slp-adia-wall"


class AdiabaticMovingWall(WallBc):
    kind = "mov-adia-wall"
    variables = ("u", "v", "w")


class IsoTNoSlipWall(IsoTWallBc, WallBc):
    kind = "no-slp-isot-wall"


class IsoTMovingWall(IsoTWallBc, WallBc):
    kind = "mov-isot-wall"
    variables = ("u", "v", "w", "T")


class ConstantVelocitySubsonicInlet(InletBc, BaseNavStokesBc):
    kind = "sub-in-fvt"
    variables = ("u", "v", "w", "T")


class SupersonicInlet(InletBc, BaseNavStokesBc):
    kind = "sup-in-fa"
    variables = ("p", "u", "v", "w", "T")


class StagnationSubsonicInlet(InletBc, BaseNavStokesBc):
    kind = "sub-in-ftptt"
    variables = ("pt", "Tt")


class ConstantMassFluxSubsonicInlet(InletBc, BaseNavStokesBc):
    kind = "sub-in-fmt"
    variables = ("mflux", "T")


class SemSubsonicInlet(SemInletBc, InletBc, BaseNavStokesBc):
    kind = "sub-in-sem"
    variables = ("T",)


class ConstantPressureSubsonicExit(BaseNavStokesBc):
    kind = "sub-out-fp"
    variables = ("p",)


class SupersonicExit(BaseNavStokesBc):
    kind = "sup-out-fn"
