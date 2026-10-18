// The dilute-gas transport of the mixture from each species' kinetic-theory
// fit in ln T: the viscosity mixed by the case's rule, the conductivity by
// the series-parallel mean. What kineticTheory imposes as it is, and what
// chungDenseGas corrects for density.
#ifndef __dilute_H__
#define __dilute_H__

#include "kernel.hpp"
#include "mixing.hpp"
#include "mixingRule.hpp"

// mu and kappa of the mixture at T, u = ln T, from the mole fractions X;
// the viscosity's fit is of sqrt(mu) / T^(1/4), the conductivity's of
// kappa / sqrt(T), Horner in u
KOKKOS_INLINE_FUNCTION void diluteTransport(const fpdtype T, const fpdtype u,
                                            const fpdtype *X, fpdtype &mu,
                                            fpdtype &kappa) {
  const fpdtype sqrtT = sqrt(T);
  const fpdtype sqrtsqrtT = sqrt(sqrtT);
  // each species' array lives only until its mixture rule has run
  {
    fpdtype sqrtMu[ns];
    for (int n = 0; n <= ns - 1; n++) {
      fpdtype fit = 0.0;
      for (int m = muPolyTerms - 1; m >= 0; m--)
        fit = fit * u + muPoly(n, m);
      sqrtMu[n] = fit * sqrtsqrtT;
    }
    mu = mixingRule::viscosity(X, sqrtMu);
  }
  fpdtype kappaSp[ns];
  for (int n = 0; n <= ns - 1; n++) {
    fpdtype fit = 0.0;
    for (int m = kappaPolyTerms - 1; m >= 0; m--)
      fit = fit * u + kappaPoly(n, m);
    kappaSp[n] = fit * sqrtT;
  }
  kappa = mixtureConductivity(X, kappaSp);
}

#endif
