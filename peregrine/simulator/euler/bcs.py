from peregrine.simulator.advec.bcs import BaseAdvecBc
from peregrine.simulator.sem.boundary import SemInletBc


class InletBc(BaseAdvecBc):
    # the carried species' mass fractions are the primitive names after p, u, v, w, T
    @classmethod
    def varnames(cls, privars):
        return super().varnames(privars) + tuple(privars[5:])


class IsoTWallBc(BaseAdvecBc):
    variables = ("T",)


class BaseEulerBc(BaseAdvecBc):
    pass


class WallBc(BaseEulerBc):
    # overwrites the flux function's flux through the wall's faces with the wall's own
    def postflux(self, Q):
        return self._kernels("postflux", faces=True, Q=Q, state="state", F="{a}F")


class AdiabaticSlipWall(WallBc):
    kind = "slp-adia-wall"


class IsoTSlipWall(IsoTWallBc, WallBc):
    kind = "slp-isot-wall"


class ConstantVelocitySubsonicInlet(InletBc, BaseEulerBc):
    kind = "sub-in-fvt"
    variables = ("u", "v", "w", "T")


class SupersonicInlet(InletBc, BaseEulerBc):
    kind = "sup-in-fa"
    variables = ("p", "u", "v", "w", "T")


class StagnationSubsonicInlet(InletBc, BaseEulerBc):
    kind = "sub-in-ftptt"
    variables = ("pt", "Tt")


class ConstantMassFluxSubsonicInlet(InletBc, BaseEulerBc):
    kind = "sub-in-fmt"
    variables = ("mflux", "T")


class SemSubsonicInlet(SemInletBc, InletBc, BaseEulerBc):
    kind = "sub-in-sem"
    variables = ("T",)


class ConstantPressureSubsonicExit(BaseEulerBc):
    kind = "sub-out-fp"
    variables = ("p",)


class SupersonicExit(BaseEulerBc):
    kind = "sup-out-fn"
