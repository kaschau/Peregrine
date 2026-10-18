// Koren's limiter: van Leer's kappa scheme at kappa = 1/3, the face value
// weighted a third toward the upwind difference and two thirds toward the
// one across the face, held within the TVD region. Third order for cell
// averages differenced on a uniform line, second on a general grid, with
// the smallest leading error of the family; upwind-biased, so not
// symmetric.
#ifndef __limiterKoren_H__
#define __limiterKoren_H__

#include "abi.hpp"

struct koren {
  static constexpr fpdtype limit = 2.0;
  static KOKKOS_INLINE_FUNCTION fpdtype phi(fpdtype r) {
    return fmax(0.0, fmin(fmin(2.0 * r, (2.0 + r) / 3.0), 2.0));
  }
};

#endif
