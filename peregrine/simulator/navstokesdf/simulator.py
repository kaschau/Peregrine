from peregrine.simulator.eulerdf import EulerDoubleFluxSimulator
from peregrine.simulator.navstokes import NavierStokesSimulator
from peregrine.simulator.navstokesdf.bcs import BaseNavStokesDfBc


class NavierStokesDoubleFluxSimulator(EulerDoubleFluxSimulator, NavierStokesSimulator):
    name = "navier-stokes-df"
    bcscls = BaseNavStokesDfBc
    # the diffusive flux's files in the planes, one per formula this extension supports
    difftpls = f"{__package__}.kernels.flux"
