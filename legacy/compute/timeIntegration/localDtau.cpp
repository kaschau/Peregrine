#include "advFlux/dissipation.hpp"
#include "array"
#include "dualTime.hpp"
#include "kernel.hpp"
#include "utils/normal.hpp"
#include "vector"

#if !defined(PG_PSEUDO_CFL) || !defined(PG_PSEUDO_VNN)
#error "localDtau takes the pseudo CFL and VNN the config names"
#endif
#if !defined(PG_PSEUDO_REAL) || !defined(PG_PSEUDO_IMAGINARY)
#error "localDtau takes the pseudo scheme's stability limits, real and imaginary"
#endif

PG_RANGE(cellCenters)
struct localDtau {
  cellVecIn Q, dIJK, qh, qt;
  iFaceStradVecIn iS;
  jFaceStradVecIn jS;
  kFaceStradVecIn kS;
  cellScalOut dtau;
  scalarIn dt;
  bool viscous;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    //-------------------------------------------------------------------------------------------|
    // Compute local pseudo time step
    //-------------------------------------------------------------------------------------------|
    // Cell lengths; an axis not marched in is infinitely long
    const fpdtype &dI = dIJK(0);
    const fpdtype &dJ = dIJK(1);
    const fpdtype &dK = dIJK(2);

    // Find max convective CFL
    fpdtype S0, S1;
    fpdtype inx0, iny0, inz0, inx1, iny1, inz1;
    faceNormal(iS.L(0), iS.L(1), iS.L(2), S0, inx0, iny0, inz0);
    faceNormal(iS.R(0), iS.R(1), iS.R(2), S1, inx1, iny1, inz1);
    fpdtype jnx0, jny0, jnz0, jnx1, jny1, jnz1;
    faceNormal(jS.L(0), jS.L(1), jS.L(2), S0, jnx0, jny0, jnz0);
    faceNormal(jS.R(0), jS.R(1), jS.R(2), S1, jnx1, jny1, jnz1);
    fpdtype knx0, kny0, knz0, knx1, kny1, knz1;
    faceNormal(kS.L(0), kS.L(1), kS.L(2), S0, knx0, kny0, knz0);
    faceNormal(kS.R(0), kS.R(1), kS.R(2), S1, knx1, kny1, knz1);
    // the velocity off the conserved state
    const fpdtype rhoinv = 1.0 / Q(0);
    const fpdtype u = Q(1) * rhoinv;
    const fpdtype v = Q(2) * rhoinv;
    const fpdtype w = Q(3) * rhoinv;

    fpdtype uI = sqrt(pow(0.5 * (inx0 + inx1) * u, 2.0) +
                      pow(0.5 * (iny0 + iny1) * v, 2.0) +
                      pow(0.5 * (inz0 + inz1) * w, 2.0));
    fpdtype uJ = sqrt(pow(0.5 * (jnx0 + jnx1) * u, 2.0) +
                      pow(0.5 * (jny0 + jny1) * v, 2.0) +
                      pow(0.5 * (jnz0 + jnz1) * w, 2.0));
    fpdtype uK = sqrt(pow(0.5 * (knx0 + knx1) * u, 2.0) +
                      pow(0.5 * (kny0 + kny1) * v, 2.0) +
                      pow(0.5 * (knz0 + knz1) * w, 2.0));

    const fpdtype &c = qh(3);

    // the pseudo step's bounds, the config's, baked in
    const fpdtype pseudoCFL = PG_PSEUDO_CFL;
    const fpdtype pseudoVNN = PG_PSEUDO_VNN;

    const fpdtype nu = viscous ? qt(0) * rhoinv : 0.0;
#ifdef PG_LOW_MACH
    // the preconditioned system's wave speeds set the pseudo step: it
    // propagates u' +- c', not u + c
    const fpdtype Ur = referenceVelocity(sqrt(u * u + v * v + w * w), c, nu,
                                         dI, dJ, dK, dt());
    const fpdtype alpha = 0.5 * (1.0 - Ur * Ur / (c * c));
#else
    // unpreconditioned, the system propagates u +- c
    const fpdtype Ur = c, alpha = 0.0;
#endif
    const fpdtype a2 = alpha * alpha;
    const fpdtype Ur2 = Ur * Ur;

    // the bound is what the fastest mode on a central scheme sees, oblique
    // to the grid: the preconditioned system's wave speed for the velocity
    // summed over the directions, each over its length, and its reference
    // velocity over the length 1 / sqrt(sum 1 / d^2); a per-direction bound
    // would let an isotropic cell's oblique mode run sqrt(3) too fast in
    // three dimensions. Diffusion sums the directions as it is. An infinite
    // length contributes nothing.
    const fpdtype uSum = uI / dI + uJ / dJ + uK / dK;
    const fpdtype invD2 = 1.0 / (dI * dI) + 1.0 / (dJ * dJ) + 1.0 / (dK * dK);
    const fpdtype waves =
        (1.0 - alpha) * uSum + sqrt(a2 * uSum * uSum + Ur2 * invD2);
    // the fluxes' numerical diffusion, a speed s along a direction being a
    // diffusion of s d / 2 with the eigenvalue 2 s / d: one that acts in
    // every direction summed over them, one along a single direction at its
    // largest; scaled so that a diffusion alone meets the pseudo scheme's
    // real-axis limit at the pseudoCFL a wave alone meets its imaginary one
    // (sqrt(3) and 2.51 for rk3), both baked from the scheme's stages. A
    // scheme with no imaginary-axis limit has no wave bound, and its
    // real-axis one stands in.
    constexpr fpdtype realLimit = PG_PSEUDO_REAL;
    constexpr fpdtype imaginaryLimit =
        PG_PSEUDO_IMAGINARY > 0.0 ? PG_PSEUDO_IMAGINARY : PG_PSEUDO_REAL;
    fpdtype summing = 0.0, alone = 0.0;
    const fpdtype us[] = {uI, uJ, uK}, ds[] = {dI, dJ, dK};
    for (int d = 0; d < 3; d++) {
      fpdtype s, a;
      numericalDiffusion::along(us[d], c, Ur, alpha, s, a);
      summing += 2.0 * s / ds[d];
      alone = fmax(alone, 2.0 * a / ds[d]);
    }
    const fpdtype diffusion = (imaginaryLimit / realLimit) * (summing + alone);
    // waves are imaginary eigenvalues and a diffusion real ones, and the
    // pseudo scheme's stability region holds their combination in
    // quadrature, so the two rates combine that way rather than add
    fpdtype rate = sqrt(waves * waves + diffusion * diffusion) / pseudoCFL;
    if (viscous) {
      rate += nu * invD2 / pseudoVNN;
    }

    dtau = 1.0 / rate;
  }
};

PG_ABI void pgLocalDtau(const localDtau &k, const pgTiling &t) {
  forCells("localDtau", t, k);
}
