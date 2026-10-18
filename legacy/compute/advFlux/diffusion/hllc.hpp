// The numerical diffusion hllc adds: at most the plain largest wave speed's,
// on every variable in every direction; its contact wave carries none.
#ifndef __diffusionHllc_H__
#define __diffusionHllc_H__

#include "kernel.hpp"

struct hllcDiffusion {
  static KOKKOS_INLINE_FUNCTION void along(const fpdtype un, const fpdtype c,
                                           const fpdtype, const fpdtype,
                                           fpdtype &summing, fpdtype &alone) {
    summing = fabs(un) + c;
    alone = 0.0;
  }
};

#endif
