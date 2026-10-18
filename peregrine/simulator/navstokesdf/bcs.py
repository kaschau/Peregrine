from peregrine.simulator.advec.bcs import BaseAdvecBc
from peregrine.simulator.euler.bcs import InletBc, IsoTWallBc
from peregrine.simulator.navstokes.bcs import GradientHooks, WallHooks
from peregrine.simulator.sem.boundary import SemInletBc


class BaseNavStokesDfBc(GradientHooks, BaseAdvecBc):
    pass


class WallBc(WallHooks, BaseNavStokesDfBc):
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


class ConstantVelocitySubsonicInlet(InletBc, BaseNavStokesDfBc):
    kind = "sub-in-fvt"
    variables = ("u", "v", "w", "T")


class SupersonicInlet(InletBc, BaseNavStokesDfBc):
    kind = "sup-in-fa"
    variables = ("p", "u", "v", "w", "T")


class StagnationSubsonicInlet(InletBc, BaseNavStokesDfBc):
    kind = "sub-in-ftptt"
    variables = ("pt", "Tt")


class ConstantMassFluxSubsonicInlet(InletBc, BaseNavStokesDfBc):
    kind = "sub-in-fmt"
    variables = ("mflux", "T")


class SemSubsonicInlet(SemInletBc, InletBc, BaseNavStokesDfBc):
    kind = "sub-in-sem"
    variables = ("T",)


class ConstantPressureSubsonicExit(BaseNavStokesDfBc):
    kind = "sub-out-fp"
    variables = ("p",)


class SupersonicExit(BaseNavStokesDfBc):
    kind = "sup-out-fn"
