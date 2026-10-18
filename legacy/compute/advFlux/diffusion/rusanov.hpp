// The numerical diffusion rusanov adds: the largest wave speed, the
// preconditioned |u'| + c' or, with Ur = c and alpha = 0, |u| + c, on every
// variable in every direction.
#ifndef __diffusionRusanov_H__
#define __diffusionRusanov_H__

#include "kernel.hpp"

struct rusanovDiffusion {
  static KOKKOS_INLINE_FUNCTION void along(const fpdtype un, const fpdtype,
                                           const fpdtype Ur, const fpdtype alpha,
                                           fpdtype &summing, fpdtype &alone) {
    summing = fabs((1.0 - alpha) * un) + sqrt(alpha * alpha * un * un + Ur * Ur);
    alone = 0.0;
  }
};

#endif
