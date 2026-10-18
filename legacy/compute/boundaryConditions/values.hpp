// What a boundary entry gives a condition. A condition's header names its
// variables in one enum, `enum variable { u, v, w, T };`, and a body that
// reads them holds the entry, `bcEntry entry`, and the clock, `scalarIn tme`:
// `const auto given = entry.at(tme())` is the entry at the clock's time,
// given(k) variable k and given.Y() the mass fractions, as a callable of the
// species, the last from the rest. The jit bakes the entry into the kernel:
// PG_BC_VALUES is one {low, high} pair per variable and then per species but
// the last, both the same for a constant, a ramp's two ends otherwise;
// PG_BC_NVARS counts the variables; a ramp's clock is PG_RAMP_START and
// PG_RAMP_INV_DURATION, and PG_RAMP the shape whose fraction(s) is the way
// along it. The jit forces in one of two forms of bcEntry, values/baked.hpp
// -- the numbers themselves -- or values/profile.hpp -- a plane over the
// face the numbers multiply.
#ifndef __values_H__
#define __values_H__

#include "kernel.hpp"

// one value of an entry: its two ends, and where between them the clock
// puts it; a constant folds, whatever the clock does
struct value {
  fpdtype low, high;
  KOKKOS_INLINE_FUNCTION constexpr fpdtype at(const fpdtype f) const {
    return low == high ? low : low + (high - low) * f;
  }
};

// the way along the entry's ramp at time t, clipped to its ends; one, for
// an entry that does not ramp
KOKKOS_INLINE_FUNCTION constexpr fpdtype rampFraction(const fpdtype t) {
#ifdef PG_RAMP
  const fpdtype s = (t - PG_RAMP_START) * PG_RAMP_INV_DURATION;
  return PG_RAMP::fraction(s < 0.0 ? 0.0 : (s > 1.0 ? 1.0 : s));
#else
  return 1.0;
#endif
}

// an entry at one time, which a body reads its values through: variable k
// by index, and the mass fractions as a callable of the species -- the
// values past the variables, the last species from the rest
template <class E> struct entryAt {
  E entry;
  fpdtype f;
  KOKKOS_INLINE_FUNCTION fpdtype operator()(const int k) const {
    return entry.read(k, f);
  }
  KOKKOS_INLINE_FUNCTION auto Y() const {
    return [e = entry, f = f](const int n) {
      if (n < ns - 1) {
        return e.read(PG_BC_NVARS + n, f);
      }
      fpdtype last = 1.0;
      for (int m = 0; m < ns - 1; m++) {
        last -= e.read(PG_BC_NVARS + m, f);
      }
      return last;
    };
  }
};

#endif
