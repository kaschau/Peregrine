// Rusanov: the average of the two sides' fluxes, less the largest wave
// speed times the jump. Under low-Mach preconditioning (PG_LOW_MACH) the
// dissipation is the preconditioned system's: its largest eigenvalue,
// |u'| + c' of Weiss and Smith, times Gamma's product with the jump in the
// primitives, which is the conserved jump plus (1 / Ur^2 - 1 / c^2) times
// the pressure jump along (1, u, v, w, H, Y); with Ur = c it is the plain
// scheme.
#ifndef __formulaRusanov_H__
#define __formulaRusanov_H__

#include "advFlux/faceState.hpp"
#include "faces.hpp"
#include "thermo/doubleFlux.hpp"
#include "timeIntegration/dualTime.hpp"
#include "utils/normal.hpp"

struct rusanov {
  template <class Recon, class Out>
  static KOKKOS_INLINE_FUNCTION void flux(const Recon &r, const faceVecIn &A,
                                          const Out &F) {
    fpdtype S, nx, ny, nz;
    faceNormal(A(0), A(1), A(2), S, nx, ny, nz);
    const auto s = r.states();
    const faceState &L = s.L, &R = s.R;
    fpdtype UR = normalVelocity(R.rho, R.rhou, R.rhov, R.rhow, nx, ny, nz);
    fpdtype UL = normalVelocity(L.rho, L.rhou, L.rhov, L.rhow, nx, ny, nz);

    // the face's state for the dissipation: the mean of the two sides
    const fpdtype rhoinvL = 1.0 / L.rho, rhoinvR = 1.0 / R.rho;
    const fpdtype u = 0.5 * (L.rhou * rhoinvL + R.rhou * rhoinvR);
    const fpdtype v = 0.5 * (L.rhov * rhoinvL + R.rhov * rhoinvR);
    const fpdtype w = 0.5 * (L.rhow * rhoinvL + R.rhow * rhoinvR);
    const fpdtype c = 0.5 * (L.c + R.c);
#ifdef PG_LOW_MACH
    // the preconditioned eigenvalue, u' +- c' at the reference velocity of
    // the pseudo step: the flow speed floored by eps c and by the smallest
    // cell length over pi dt, as the preconditioner and the pseudo step
    // take it, an inviscid call for the diffusion floor
    const fpdtype Un = 0.5 * (UL + UR);
    const fpdtype speed = sqrt(u * u + v * v + w * w);
    const fpdtype dI = 0.5 * (r.dIJK.L(0) + r.dIJK.R(0)),
                  dJ = 0.5 * (r.dIJK.L(1) + r.dIJK.R(1)),
                  dK = 0.5 * (r.dIJK.L(2) + r.dIJK.R(2));
    const fpdtype Ur = referenceVelocity(speed, c, 0.0, dI, dJ, dK, r.dt());
    const fpdtype alpha = 0.5 * (1.0 - Ur * Ur / (c * c));
    const fpdtype lam =
        (fabs((1.0 - alpha) * Un) + sqrt(alpha * alpha * Un * Un + Ur * Ur)) *
        S;
    // Gamma's pressure column less dQ/dq's, times the pressure jump
    const fpdtype dp = (1.0 / (Ur * Ur) - 1.0 / (c * c)) * (R.p - L.p);
#else
    const fpdtype lam = fmax(fabs(UL) + R.c, fabs(UR) + L.c) * S;
    const fpdtype dp = 0.0;
#endif
    UR *= S;
    UL *= S;

    // Continuity rho*Ui
    F(0) = 0.5 * (UR * R.rho + UL * L.rho - lam * (R.rho - L.rho + dp));
    // momentum rho*u*Ui + p*A, each side's own then the average
    const fpdtype *mL[] = {&L.rhou, &L.rhov, &L.rhow},
                  *mR[] = {&R.rhou, &R.rhov, &R.rhow};
    const fpdtype vel[] = {u, v, w};
    for (int d = 0; d < 3; d++) {
      const fpdtype FUR = UR * *mR[d] + R.p * A(d);
      const fpdtype FUL = UL * *mL[d] + L.p * A(d);
      F(1 + d) = 0.5 * (FUR + FUL - lam * (*mR[d] - *mL[d] + dp * vel[d]));
    }
    // Total energy (rhoE + p)*Ui
#ifdef PG_DOUBLE_FLUX
    // both sides' energies in each cell's frozen relation: the flux for the
    // left cell in F(4), for the right in F(ne)
    const auto energyFlux = [&](const auto &qh) -> fpdtype {
      const fpdtype EL = doubleFlux::stateEnergy(r, s, false, qh);
      const fpdtype ER = doubleFlux::stateEnergy(r, s, true, qh);
      const fpdtype H = 0.5 * ((EL + L.p) * rhoinvL + (ER + R.p) * rhoinvR);
      return 0.5 * (UR * (ER + R.p) + UL * (EL + L.p) -
                    lam * (ER - EL + dp * H));
    };
    F(4) = energyFlux(r.qh.L());
    F(ne) = energyFlux(r.qh.R());
#else
    const fpdtype H = 0.5 * ((L.E + L.p) * rhoinvL + (R.E + R.p) * rhoinvR);
    F(4) = 0.5 * (UR * (R.E + R.p) + UL * (L.E + L.p) -
                  lam * (R.E - L.E + dp * H));
#endif
    // Species
    for (int n = 0; n < ne - 5; n++) {
      fpdtype rhoYL, rhoYR;
      r.species(s, n, rhoYL, rhoYR);
      const fpdtype Y = 0.5 * (rhoYL * rhoinvL + rhoYR * rhoinvR);
      F(5 + n) = 0.5 * (rhoYR * UR + rhoYL * UL - lam * (rhoYR - rhoYL + dp * Y));
    }
  }
};

#endif
