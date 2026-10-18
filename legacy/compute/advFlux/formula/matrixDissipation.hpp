// Matrix dissipation, a correction to a central flux: the Roe-type term
// -1/2 Gamma |Gamma^-1 A| dq of Swanson and Turkel, each characteristic
// field of the jump between the two sides damped at its own wave speed --
// the acoustic fields, the pressure and the normal velocity and the
// isentropic part of the temperature, at c', and the convected ones, the
// rest of the temperature, the transverse velocity and the species, at
// |u|. Under low-Mach preconditioning (PG_LOW_MACH) c' is Weiss and Smith's
// preconditioned acoustic speed and Gamma the preconditioning matrix, so
// the pseudo march sees the speeds its step is built on; the entropy,
// shear and species waves keep their eigenvalue u under it. Without it c'
// is c and Gamma is dQ/dq. c' is never below the acoustic fields' u', so
// the mean of the two acoustic speeds is c' and the cross term of the Roe
// matrix, of order u', is left out. The primitive jump is taken in
// (p, u, T, Y) between the two sides' states, the matrix at the face's
// mean state, the eos's derivatives there. Each field is weighed by the
// switch's sensor for it: the pressure and the velocity by the shock
// sensor alone, the species and the temperature by the larger of the
// shock and contact sensors -- a contact gets the species and the
// temperature wrong and leaves the pressure and the velocity alone, so a
// composition gradient does not dissipate the shear that rides on it.
// Under the double flux (PG_DOUBLE_FLUX) the energy row is each cell's
// frozen relation's, written once per side as every formula does.
#ifndef __formulaMatrixDissipation_H__
#define __formulaMatrixDissipation_H__

#include "advFlux/faceState.hpp"
#include "advFlux/fluxOut.hpp"
#include "faces.hpp"
#include "thermo/doubleFlux.hpp"
#include "thermo/eos.hpp"
#include "timeIntegration/dualTime.hpp"
#include "utils/normal.hpp"

struct matrixDissipation : correction {
  template <class Recon, class Out>
  static KOKKOS_INLINE_FUNCTION void flux(const Recon &r, const faceVecIn &A,
                                          const Out &F) {
    fpdtype S, nx, ny, nz;
    faceNormal(A(0), A(1), A(2), S, nx, ny, nz);
    const auto s = r.states();
    const faceState &L = s.L, &R = s.R;

    // the two sides' primitives and the face's mean
    const fpdtype rhoinvL = 1.0 / L.rho, rhoinvR = 1.0 / R.rho;
    const fpdtype uL[3] = {L.rhou * rhoinvL, L.rhov * rhoinvL, L.rhow * rhoinvL};
    const fpdtype uR[3] = {R.rhou * rhoinvR, R.rhov * rhoinvR, R.rhow * rhoinvR};
    const fpdtype TL = r.q.L(1), TR = r.q.R(1);
    const fpdtype rho = 0.5 * (L.rho + R.rho), rhoinv = 1.0 / rho;
    const fpdtype p = 0.5 * (L.p + R.p), T = 0.5 * (TL + TR);
    const fpdtype c = 0.5 * (L.c + R.c);
    const fpdtype u[3] = {0.5 * (uL[0] + uR[0]), 0.5 * (uL[1] + uR[1]),
                          0.5 * (uL[2] + uR[2])};
    const fpdtype un = u[0] * nx + u[1] * ny + u[2] * nz;
    const fpdtype ke = 0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    fpdtype Y[ns], dY[ns];
    for (int n = 0; n < ns - 1; n++) {
      fpdtype rhoYL, rhoYR;
      r.species(s, n, rhoYL, rhoYR);
      const fpdtype YL = rhoYL * rhoinvL, YR = rhoYR * rhoinvR;
      Y[n] = 0.5 * (YL + YR);
      dY[n] = YR - YL;
    }
    Y[ns - 1] = 1.0;
    for (int n = 0; n < ns - 1; n++)
      Y[ns - 1] -= Y[n];

    // the wave speeds, preconditioned or plain
#ifdef PG_LOW_MACH
    const fpdtype speed = sqrt(2.0 * ke);
    const fpdtype dI = 0.5 * (r.dIJK.L(0) + r.dIJK.R(0)),
                  dJ = 0.5 * (r.dIJK.L(1) + r.dIJK.R(1)),
                  dK = 0.5 * (r.dIJK.L(2) + r.dIJK.R(2));
    const fpdtype Ur = referenceVelocity(speed, c, 0.0, dI, dJ, dK, r.dt());
    const fpdtype alpha = 0.5 * (1.0 - Ur * Ur / (c * c));
#else
    const fpdtype Ur = c, alpha = 0.0;
#endif
    const fpdtype convective = fabs(un);
    const fpdtype acoustic = sqrt(alpha * alpha * un * un + Ur * Ur);

    // the density's derivatives at the face, and the preconditioning matrix's
    // pressure entry Theta relaxing to rho_p through the compressibility
    fpdtype rho_p, rho_T, rho_Y[ns], beta;
#ifdef PG_DOUBLE_FLUX
    // the frozen relation's, in the scaled temperature: the eos's speed of
    // sound is its compressibility
    rho_p = rho / p;
    rho_T = -rho / T;
    for (int n = 0; n < ns - 1; n++)
      rho_Y[n] = 0.0;
    beta = 1.0 / (c * c);
#else
    eos::densityDerivatives(
        p, T, rho, [&](const int n) { return Y[n]; }, rho_p, rho_T, rho_Y);
    // the isentropic compressibility, as the pseudo Jacobian takes it
    beta = 1.0 / (c * c);
#endif
#ifdef PG_LOW_MACH
    const fpdtype Theta = 1.0 / (Ur * Ur) + rho_p - beta;
#else
    const fpdtype Theta = rho_p;
    (void)beta;
#endif

    // the primitive jump, each field at its speed and by its sensor: the
    // pressure and the normal velocity acoustic, the transverse velocity
    // and the species convective, the temperature convective but for its
    // isentropic part with the pressure, dT/dp at fixed entropy being
    // -T rho_T / (rho^2 cp)
    const fpdtype ws = r.shockWeight(), wc = r.contactWeight();
    const fpdtype dp = ws * acoustic * (R.p - L.p);
    const fpdtype dun = (uR[0] - uL[0]) * nx + (uR[1] - uL[1]) * ny +
                        (uR[2] - uL[2]) * nz;
    const fpdtype dus[3] = {uR[0] - uL[0], uR[1] - uL[1], uR[2] - uL[2]};
    fpdtype du[3];
    du[0] = ws * (convective * dus[0] + (acoustic - convective) * dun * nx);
    du[1] = ws * (convective * dus[1] + (acoustic - convective) * dun * ny);
    du[2] = ws * (convective * dus[2] + (acoustic - convective) * dun * nz);
    fpdtype dYs[ns];
    for (int n = 0; n < ns - 1; n++)
      dYs[n] = wc * convective * dY[n];

    // the energy row's entries and the temperature jump's isentropic part
    // are each cell's relation's under the double flux, the eos's without
    const auto write = [&](const fpdtype cp, const fpdtype H, const fpdtype sE,
                           const auto &dh, const fpdtype dT, const int energy) {
      // Gamma times the damped jump: the density first, the rest on it
      fpdtype drho = Theta * dp + rho_T * dT;
      for (int n = 0; n < ns - 1; n++)
        drho += rho_Y[n] * dYs[n];
      fpdtype dE = (Theta * H + sE) * dp +
                   rho * (u[0] * du[0] + u[1] * du[1] + u[2] * du[2]) +
                   (rho_T * H + rho * cp) * dT;
      for (int n = 0; n < ns - 1; n++)
        dE += (H * rho_Y[n] + rho * dh(n)) * dYs[n];
      if (energy == 4) {
        F(0) = -0.5 * S * drho;
        F(1) = -0.5 * S * (drho * u[0] + rho * du[0]);
        F(2) = -0.5 * S * (drho * u[1] + rho * du[1]);
        F(3) = -0.5 * S * (drho * u[2] + rho * du[2]);
        for (int n = 0; n < ns - 1; n++)
          F(5 + n) = -0.5 * S * (drho * Y[n] + rho * dYs[n]);
      }
      F(energy) = -0.5 * S * dE;
    };
    const fpdtype dTraw = TR - TL;
#ifdef PG_DOUBLE_FLUX
    // in one cell's relation, into its energy slot
    const auto sided = [&](const auto &qh, const int energy) {
      const fpdtype theta = qh(doubleFlux::slotTheta),
                    kappa = qh(doubleFlux::slotKappa);
      const fpdtype cp = ((theta + 1.0) * p + kappa) / (rho * T);
      fpdtype zetaY = 0.0;
      for (int n = 0; n < ns - 1; n++)
        zetaY += qh(doubleFlux::slotZeta + n) * Y[n];
      // the conserved energy's total enthalpy in the relation, kappa
      // included, as the pseudo Jacobian has it
      const fpdtype H = qh(doubleFlux::slotEps) + zetaY + ke +
                        ((theta + 1.0) * p + kappa) * rhoinv;
      const fpdtype sE = -1.0 - kappa / p;
      const fpdtype dTac = -T * rho_T * (R.p - L.p) / (rho * rho * cp);
      const fpdtype dT =
          wc * convective * dTraw + ws * (acoustic - convective) * dTac;
      write(
          cp, H, sE, [&](const int n) { return qh(doubleFlux::slotZeta + n); },
          dT, energy);
    };
    sided(r.qh.L(), 4);
    sided(r.qh.R(), ne);
#else
    const fpdtype cp = 0.5 * (r.qh.L(1) + r.qh.R(1));
    const fpdtype H = 0.5 * (r.qh.L(2) * rhoinvL + r.qh.R(2) * rhoinvR) + ke;
    const fpdtype sE = T * rho_T * rhoinv;
    const auto hL = eos::enthalpies(TL, r.qh.L());
    const auto hR = eos::enthalpies(TR, r.qh.R());
    const fpdtype dTac = -T * rho_T * (R.p - L.p) / (rho * rho * cp);
    const fpdtype dT =
        wc * convective * dTraw + ws * (acoustic - convective) * dTac;
    write(
        cp, H, sE,
        [&](const int n) {
          return 0.5 * (hL(n) + hR(n)) - 0.5 * (hL(ns - 1) + hR(ns - 1));
        },
        dT, 4);
#endif
  }
};

#endif
