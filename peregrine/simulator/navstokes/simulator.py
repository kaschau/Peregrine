from peregrine.config import ConfigError
from peregrine.launch.kernel import Kernel
from peregrine.launch.ranges import CellCenterRange
from peregrine.simulator.advecdiff import BaseAdvecDiffSimulator
from peregrine.simulator.euler import EulerSimulator
from peregrine.simulator.navstokes.bcs import BaseNavStokesBc

templates = f"{__package__}.kernels"


class NavierStokesSimulator(EulerSimulator, BaseAdvecDiffSimulator):
    name = "navier-stokes"
    bcscls = BaseNavStokesBc
    # the package of the diffusive flux's formula files
    difftpls = f"{__package__}.kernels.flux"

    def validate(self):
        super().validate()
        if self.cfg["mixture"]["trans"] is None:
            msg = "a viscous case names a transport model"
            raise ConfigError("mixture.trans", msg)

    @property
    def ngrad(self):
        # every primitive but the pressure
        return self.ne - 1

    def arrays(self):
        arrays = super().arrays()
        ns = self.mixture.ns
        # trstate is mu, kappa, then a diffusivity per species
        arrays["trstate"] = dict(kind=CellCenterRange, comps=2 + ns)
        # hsp is each species' enthalpy at the cell, read by the diffusive flux
        arrays["hsp"] = dict(kind=CellCenterRange, comps=ns)
        return arrays

    def tplargs(self):
        mixture = self.mixture
        return dict(
            super().tplargs(),
            ngrad=self.ngrad,
            trans=mixture.trans.template(),
            diffusion=mixture.diffusion.template(),
            mixrule=mixture.mixrule.template(),
        )

    @property
    def fluxbinds(self):
        # the ducros switch reads the gradients
        return dict(super().fluxbinds, grads="grads")

    def decl_kerns(self, provider):
        trans = f"{templates}.transport"
        grad = f"{templates}.dqdxyz"
        k = super().decl_kerns(provider)
        k["trans"] = lambda Q: Kernel(provider, trans, Q=Q, state="state", trstate="trstate", hsp="hsp")
        cells = dict(state="state", Jinv="Jinv", grads="grads")
        k["dqdxyz"] = lambda Q: Kernel(provider, grad, Q=Q, **cells)
        # the diffusive flux: the formula's own file, the face centers per axis
        cfgsect = self.cfg["diff-flux"]
        diffflux = f"{self.difftpls}.{cfgsect['formula']}"
        faces = dict(state="state", trstate="trstate", hsp="hsp", grads="grads")
        faces.update(xcc="xcc", xf="{a}xc")
        k["diffflux"] = lambda Q: self.facekernels(diffflux, cfgsect, Q=Q, **faces)
        return k
