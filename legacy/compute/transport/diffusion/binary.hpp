// Mixture-averaged diffusion from the binary coefficients, each pair's a
// fit in ln T from the collision integrals.
#ifndef __binary_H__
#define __binary_H__

#include "transport/mixing.hpp"

namespace binary {

// each species' D from the binary fits, Horner in u = ln T at unit pressure
// and scaled by T^(3/2), each pair's times factor(n, m), one for the dilute
// gas: D[n] = 1 / (p sum_m X_m / D_nm + p X_n / (MWmix - MW_n X_n) sum_m X_m
// MW_m / D_nm), the sums over m != n. Each pair's fit is evaluated twice,
// once from each side, so a species' two sums are scalars: a per-species
// array of sums is indexed by the inner loop, which puts it in memory, and
// that cost three times the arithmetic it saved. MWmix - MW_n X_n is taken
// as its own sum, sum_m X_m MW_m: near a pure species the difference cancels
// to rounding. A pure fluid has no diffusion coefficient: every D is zero.
// Pure is no species beside one, not X_n == 1: a mass fraction off the
// conserved ratio lands a rounding either side of one, and with the others'
// clipped to zero the sums are zero and D is 1 / 0.
template <class Factor>
KOKKOS_INLINE_FUNCTION void mixtureAveraged(const mixtureState &s, fpdtype *D,
                                            const Factor &factor) {
  const fpdtype p = s.p, u = s.u;
  const fpdtype *X = s.X;
  for (int n = 0; n <= ns - 1; n++) {
    fpdtype others = 0.0;
    for (int m = 0; m <= ns - 1; m++) {
      if (m != n) {
        others += X[m];
      }
    }
    if (!(others > 0.0)) {
      for (int m = 0; m <= ns - 1; m++) {
        D[m] = 0.0;
      }
      return;
    }
  }
  const fpdtype T_3o2 = s.T * sqrt(s.T);
  for (int n = 0; n <= ns - 1; n++) {
    fpdtype sum1 = 0.0, sum2 = 0.0, rest = 0.0;
    for (int m = 0; m <= ns - 1; m++) {
      if (m == n) {
        continue;
      }
      fpdtype Dnm = 0.0;
      for (int k = dijTerms - 1; k >= 0; k--)
        Dnm = Dnm * u + dij(n, m, k);
      const fpdtype Dinv = 1.0 / (Dnm * T_3o2 * factor(n, m));
      sum1 += X[m] * Dinv;
      sum2 += X[m] * MW(m) * Dinv;
      rest += X[m] * MW(m);
    }
    D[n] = 1.0 / (p * sum1 + p * X[n] / rest * sum2);
  }
}

KOKKOS_INLINE_FUNCTION void coefficients(const mixtureState &s, fpdtype *D) {
  mixtureAveraged(s, D, [](const int, const int) { return fpdtype(1.0); });
}

} // namespace binary

#endif
