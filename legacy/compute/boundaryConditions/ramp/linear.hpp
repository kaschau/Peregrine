// The shapes a boundary value ramps by: the way from one end to the other
// as a function of the way through the duration, 0 to 1.
#ifndef __ramp_linear_H__
#define __ramp_linear_H__

#include "kernel.hpp"

namespace ramp {

struct linear {
  static KOKKOS_INLINE_FUNCTION constexpr fpdtype fraction(const fpdtype s) {
    return s;
  }
};

} // namespace ramp

#endif
