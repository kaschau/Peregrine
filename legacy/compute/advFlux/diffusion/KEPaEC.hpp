// The numerical diffusion KEPaEC adds: none, a central flux.
#ifndef __diffusionKEPaEC_H__
#define __diffusionKEPaEC_H__

#include "kernel.hpp"

struct KEPaECDiffusion {
  static KOKKOS_INLINE_FUNCTION void along(const fpdtype, const fpdtype,
                                           const fpdtype, const fpdtype,
                                           fpdtype &summing, fpdtype &alone) {
    summing = alone = 0.0;
  }
};

#endif
