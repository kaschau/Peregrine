// An entry's values as the numbers themselves, the kernel's constants.
#ifndef __values_baked_H__
#define __values_baked_H__

#include "boundaryConditions/values.hpp"

struct bcEntry {
  // the table is a local of the read, not a static member: nvcc keeps a
  // static member array indexed at run time on the host, out of device
  // code's reach
  KOKKOS_INLINE_FUNCTION fpdtype read(const int k, const fpdtype f) const {
    constexpr value table[] = {PG_BC_VALUES};
    return table[k].at(f);
  }
  KOKKOS_INLINE_FUNCTION entryAt<bcEntry> at(const fpdtype t) const {
    return {*this, rampFraction(t)};
  }
};

#endif
