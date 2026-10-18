// AUSM+-up: a mass flux from the two sides' Mach numbers, a pressure flux
// from their pressures, and the upwind side carrying the rest. Liou's
// scaling fa of the cutoff Mach number sizes the pressure diffusion in the
// mass flux and the velocity diffusion in the pressure flux to the flow
// speed at low Mach; what it is, its constants and the diffusion it makes
// are advFlux/diffusion/ausmPlusUp.hpp's, which the pseudo step shares.
// Under low-Mach preconditioning the cutoff is the preconditioner's
// reference Mach, so the dissipation scales with the pseudo system's wave
// speed, which a fixed cutoff, a fixed fraction of the speed of sound, does
// not.
#ifndef __formulaAusmPlusUp_H__
#define __formulaAusmPlusUp_H__

#include "advFlux/diffusion/ausmPlusUp.hpp"
#include "advFlux/faceState.hpp"
#include "faces.hpp"
#include "thermo/doubleFlux.hpp"
#include "timeIntegration/dualTime.hpp"
#include "utils/normal.hpp"

struct ausmPlusUp {
  using constants = ausmPlusUpDiffusion;
  template <class Recon, class Out>
  static KOKKOS_INLINE_FUNCTION void flux(const Recon &r, const faceVecIn &A,
                                          const Out &F) {
    fpdtype S, nx, ny, nz;
    faceNormal(A(0), A(1), A(2), S, nx, ny, nz);
    const auto s = r.states();
    const faceState &L = s.L, &R = s.R;
    const fpdtype UR =
        normalVelocity(R.rho, R.rhou, R.rhov, R.rhow, nx, ny, nz);
    const fpdtype UL =
        normalVelocity(L.rho, L.rhou, L.rhov, L.rhow, nx, ny, nz);

    const fpdtype a12 = 0.5 * (R.c + L.c);
    const fpdtype ML = UL / a12;
    const fpdtype MR = UR / a12;

    const fpdtype MbarSQ =
        (pow(UR, 2.0) + pow(UL, 2.0)) / (2.0 * pow(a12, 2.0));

#ifdef PG_LOW_MACH
    // the reference velocity of the pseudo step, from the face's mean state
    // and cell lengths, an inviscid call
    const fpdtype rhoinvL = 1.0 / L.rho, rhoinvR = 1.0 / R.rho;
    const fpdtype u = 0.5 * (L.rhou * rhoinvL + R.rhou * rhoinvR);
    const fpdtype v = 0.5 * (L.rhov * rhoinvL + R.rhov * rhoinvR);
    const fpdtype w = 0.5 * (L.rhow * rhoinvL + R.rhow * rhoinvR);
    const fpdtype speed = sqrt(u * u + v * v + w * w);
    const fpdtype dI = 0.5 * (r.dIJK.L(0) + r.dIJK.R(0)),
                  dJ = 0.5 * (r.dIJK.L(1) + r.dIJK.R(1)),
                  dK = 0.5 * (r.dIJK.L(2) + r.dIJK.R(2));
    const fpdtype Ur = referenceVelocity(speed, a12, 0.0, dI, dJ, dK, r.dt());
#else
    const fpdtype Ur = a12;
#endif
    const fpdtype fa = constants::scaling(MbarSQ, a12, Ur);

    const fpdtype beta = 1.0 / 8.0;

    const fpdtype M1Plus = 0.5 * (ML + fabs(ML));
    const fpdtype M1Minus = 0.5 * (MR - fabs(MR));
    const fpdtype M2Plus = 0.25 * pow(ML + 1.0, 2.0);
    const fpdtype M2Minus = -0.25 * pow(MR - 1.0, 2.0);
    const fpdtype M4Plus =
        (fabs(ML) >= 1.0) ? M1Plus : M2Plus * (1.0 - 16.0 * beta * M2Minus);
    const fpdtype M4Minus =
        (fabs(MR) >= 1.0) ? M1Minus : M2Minus * (1.0 + 16.0 * beta * M2Plus);

    const fpdtype rho12 = 0.5 * (R.rho + L.rho);

    const fpdtype M12 = M4Plus + M4Minus -
                        constants::Kp / fa * (fmax(1.0 - constants::sigma * MbarSQ, 0.0)) *
                            (R.p - L.p) / (rho12 * pow(a12, 2.0));

    const fpdtype mDot12 = (M12 > 0.0) ? a12 * M12 * L.rho : a12 * M12 * R.rho;

    const fpdtype alpha = 3.0 / 16.0 * (-4.0 + 5.0 * pow(fa, 2.0));
    const fpdtype p5Plus =
        (fabs(ML) >= 1.0) ? 1.0 / ML * M1Plus
                          : M2Plus * ((2.0 - ML) - 16.0 * alpha * ML * M2Minus);
    const fpdtype p5Minus =
        (fabs(MR) >= 1.0)
            ? 1.0 / MR * M1Minus
            : M2Minus * ((-2.0 - MR) + 16.0 * alpha * MR * M2Plus);

    const fpdtype p12 =
        p5Plus * L.p + p5Minus * R.p -
        constants::Ku * p5Plus * p5Minus * (R.rho + L.rho) * (fa * a12) * (UR - UL);

    // the upwind side carries the rest
    const bool fromL = mDot12 > 0.0;
    const faceState &U = fromL ? L : R;
    const fpdtype rhoinvU = 1.0 / U.rho;
    // Continuity rho*Ui
    F(0) = mDot12 * S;
    // momentum rho*u*Ui + p*A
    F(1) = mDot12 * U.rhou * rhoinvU * S + p12 * A(0);
    F(2) = mDot12 * U.rhov * rhoinvU * S + p12 * A(1);
    F(3) = mDot12 * U.rhow * rhoinvU * S + p12 * A(2);
    // Total energy (rhoE + p)*Ui
#ifdef PG_DOUBLE_FLUX
    // the upwind side's energy in each cell's frozen relation: the flux for
    // the left cell in F(4), for the right in F(ne)
    F(4) = mDot12 * (doubleFlux::stateEnergy(r, s, !fromL, r.qh.L()) + U.p) *
           rhoinvU * S;
    F(ne) = mDot12 * (doubleFlux::stateEnergy(r, s, !fromL, r.qh.R()) + U.p) *
            rhoinvU * S;
#else
    F(4) = mDot12 * (U.E + U.p) * rhoinvU * S;
#endif
    // Species
    for (int n = 0; n < ne - 5; n++) {
      fpdtype rhoYL, rhoYR;
      r.species(s, n, rhoYL, rhoYR);
      F(5 + n) = mDot12 * (fromL ? rhoYL : rhoYR) * rhoinvU * S;
    }
  }
};

#endif
