// What AUSM+-up's dissipation is made of, shared by the flux and the pseudo
// step: its constants, Liou's scaling fa of the cutoff Mach number, and the
// diffusion it adds along a direction. Under low-Mach preconditioning
// (PG_LOW_MACH) the cutoff is the preconditioner's reference Mach, never
// below the flow's own; else the mean Mach number floored at a fixed cutoff.
#ifndef __diffusionAusmPlusUp_H__
#define __diffusionAusmPlusUp_H__

#include "kernel.hpp"

struct ausmPlusUpDiffusion {
  static constexpr fpdtype Kp = 0.25, Ku = 0.75, sigma = 1.0;

  static KOKKOS_INLINE_FUNCTION fpdtype scaling(const fpdtype MbarSQ,
                                                const fpdtype a12,
                                                const fpdtype Ur) {
#ifdef PG_LOW_MACH
    const fpdtype Mo = fmin(1.0, Ur / a12);
#else
    const fpdtype MinfSQ = 0.1;
    const fpdtype Mo = sqrt(fmin(1.0, fmax(MbarSQ, MinfSQ)));
#endif
    return Mo * (2.0 - Mo);
  }

  // the velocity diffusion of the pressure flux, Ku P5+ P5- (rho_L + rho_R)
  // fa a12 on the normal velocity, a quarter for the P5 product at low Mach
  // and twice the density for the sum, so Ku fa a12 as a speed on the
  // momentum, along its own direction alone; and the pressure diffusion of
  // the mass flux, Kp / fa dp / a12 on the pressure, which moves the density
  // the preconditioner sees, dp / Ur^2, in every direction
  static KOKKOS_INLINE_FUNCTION void along(const fpdtype un, const fpdtype c,
                                           const fpdtype Ur, const fpdtype,
                                           fpdtype &summing, fpdtype &alone) {
    const fpdtype fa = scaling(un * un / (c * c), c, Ur);
    summing = 2.0 * Kp * Ur * Ur / (fa * c);
    alone = Ku * fa * c;
  }
};

#endif
