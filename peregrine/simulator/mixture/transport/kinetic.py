import numpy as np

from peregrine.simulator.mixture.caloric.tdepcp.model import TempDepCpModel
from peregrine.simulator.mixture.constants import Ru, Tref, avogadro, kB
from peregrine.simulator.mixture.constants import epsilon0 as eps0
from peregrine.simulator.mixture.polyfit import PolyFitMixin
from peregrine.simulator.mixture.species import Species
from peregrine.simulator.mixture.transport.base import BaseTransportModel
from peregrine.simulator.mixture.transport.integrals import CollisionIntegrals


class KineticTheoryModel(BaseTransportModel, PolyFitMixin):
    name = "kinetic-theory"

    def __init__(self, cfgsect):
        super().__init__(cfgsect)
        self.needs += (
            "well", "diam", "dipole", "polarize", "zrot", "geometry"
        )
        # the conductivity needs cp against temperature, which this caloric puts on the species
        self.deps = (TempDepCpModel,)
        # ordered: the conductivity divides by the viscosity
        self.provides.update(mupoly="mupoly", kappapoly="kappapoly")

    def constants(self, species):
        mu = self.collect_poly("mupoly", species)
        kappa = self.collect_poly("kappapoly", species)
        return dict(super().constants(species), mupoly=mu, kappapoly=kappa)

    def ljparams(self, species):
        Ts = np.linspace(*self.cfgsect["T-range"], 50)
        ci = CollisionIntegrals()

        MW = self.collect("MW", species)
        well = self.collect("well", species)
        diam = self.collect("diam", species)
        dipole = self.collect("dipole", species)
        polarize = self.collect("polarize", species)
        geometry = self.collect("geometry", species)
        rotdof = np.array([Species.rotdof[g] for g in geometry])
        mass = MW / avogadro

        # every pair's reduced mass, diameter, well and dipole
        rmass = np.outer(mass, mass) / np.add.outer(mass, mass)
        rdiam = 0.5 * np.add.outer(diam, diam)
        rwell = np.sqrt(np.outer(well, well))
        rdipole = np.sqrt(np.outer(dipole, dipole))
        rdelta = 0.5 * rdipole**2 / (4 * np.pi * eps0 * rwell * rdiam**3)

        # a polar-nonpolar pair: induction deepens the well and shrinks the diameter
        polar = dipole > 0.0
        mixed = polar[:, None] != polar[None, :]
        # for each pair, the polar member's and the nonpolar member's properties
        idx = np.arange(len(MW))
        ipolar = np.where(polar[:, None], idx[:, None], idx[None, :])
        inonpolar = np.where(polar[:, None], idx[None, :], idx[:, None])
        dipolep, wellp, diamp = dipole[ipolar], well[ipolar], diam[ipolar]
        welln = well[inonpolar]
        alphastar = polarize[inonpolar] / diam[inonpolar] ** 3
        dipolestar = dipolep / np.sqrt(4 * np.pi * eps0 * diamp**3 * wellp)
        xi = 1.0 + 0.25 * alphastar * dipolestar**2 * np.sqrt(wellp / welln)
        rwell = np.where(mixed, rwell * xi**2, rwell)
        rdiam = np.where(mixed, rdiam * xi ** (-1 / 6), rdiam)

        return dict(
            Ts=Ts,
            ci=ci,
            MW=MW,
            well=well,
            diam=diam,
            mass=mass,
            zrot=self.collect("zrot", species),
            rotdof=rotdof,
            rmass=rmass,
            rdiam=rdiam,
            rwell=rwell,
            rdelta=rdelta,
        )

    def mupoly(self, species):
        m = self.ljparams(species)
        Ts, well, diam, mass = m["Ts"], m["well"], m["diam"], m["mass"]

        # (T, species): reduced temperature, and the collision integral there
        Tstar = np.outer(Ts, kB / well)
        delta = np.broadcast_to(m["rdelta"].diagonal(), Tstar.shape)
        omega22 = m["ci"].omega22(Tstar, delta)
        xsec = np.pi * diam**2 * omega22
        visc = (5.0 / 16.0) * np.sqrt(np.pi * mass * kB * Ts[:, None]) / xsec

        # the fit is of sqrt(mu) / T^(1/4); the kernel scales it back to sqrt(mu) for the mixing rule
        visc = np.sqrt(visc / np.sqrt(Ts)[:, None])
        tol, deg = self.cfgsect["refit-tol"], self.cfgsect["refit-max-deg"]
        return [self.fit(np.log(Ts), v, tol, deg)[0] for v in visc.T]

    def kappapoly(self, species):
        m = self.ljparams(species)
        Ts, well, diam, MW = m["Ts"], m["well"], m["diam"], m["MW"]
        zrot, rotdof, rmass = m["zrot"], m["rotdof"], m["rmass"].diagonal()
        lnT = np.log(Ts)

        # the species already carry mupoly and cppoly; undo their fitted forms
        sps = species.values()
        visc = np.array([self.evaluate(sp["mupoly"], lnT) for sp in sps]).T
        visc = visc**2 * np.sqrt(Ts)[:, None]
        cpR = np.array([self.evaluate(sp["cppoly"], lnT) for sp in sps]).T

        # (T, species) throughout
        T = Ts[:, None]
        Tstar = np.outer(Ts, kB / well)
        delta = np.broadcast_to(m["rdelta"].diagonal(), Tstar.shape)
        omega22 = m["ci"].omega22(Tstar, delta)
        omega11 = omega22 / m["ci"].astar(Tstar, delta)

        # self diffusion, and the Parker rotational relaxation at T and at Tref
        kT = kB * T
        xsec = np.pi * diam**2 * omega11
        Dself = (3.0 / 16.0) * np.sqrt(2.0 * np.pi / rmass) * kT**1.5 / xsec

        def fz(Tstar):
            return (
                1.0
                + np.pi**1.5 / np.sqrt(Tstar) * (0.5 + 1.0 / Tstar)
                + (0.25 * np.pi**2 + 2) / Tstar
            )

        fint = MW / (Ru * T) * Dself / visc
        cvrot = rotdof
        A = 2.5 - fint
        zrotT = zrot * fz(kB * Tref / well) / fz(Tstar)
        B = zrotT + 2.0 / np.pi * (5 / 3 * rotdof + fint)
        c1 = 2.0 / np.pi * A / B
        cvint = cpR - 2.5 - cvrot
        frot = fint * (1.0 + c1)
        ftrans = 2.5 * (1.0 - c1 * cvrot / 1.5)
        cond = visc / MW * Ru * (ftrans * 1.5 + frot * cvrot + fint * cvint)

        # the fit is of kappa / sqrt(T); the kernel scales it back
        cond = cond / np.sqrt(Ts)[:, None]
        tol, deg = self.cfgsect["refit-tol"], self.cfgsect["refit-max-deg"]
        return [self.fit(lnT, c, tol, deg)[0] for c in cond.T]
