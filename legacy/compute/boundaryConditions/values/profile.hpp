// An entry's values off a plane over the face, one per variable and then
// per species but the last, in the header's order; the baked pairs are the
// scales on them, one and one but for a scaled variable.
#ifndef __values_profile_H__
#define __values_profile_H__

#include "boundaryConditions/values.hpp"

struct bcEntry {
  blockFaceIn bcVals;
  // the scales are a local of the read, not a static member: nvcc keeps a
  // static member array indexed at run time on the host, out of device
  // code's reach
  KOKKOS_INLINE_FUNCTION fpdtype read(const int k, const fpdtype f) const {
    constexpr value table[] = {PG_BC_VALUES};
    return bcVals(k) * table[k].at(f);
  }
  KOKKOS_INLINE_FUNCTION entryAt<bcEntry> at(const fpdtype t) const {
    return {*this, rampFraction(t)};
  }
  template <class P> KOKKOS_INLINE_FUNCTION void pin(const P &p) {
    bcVals.pin(p);
  }
};

#endif
