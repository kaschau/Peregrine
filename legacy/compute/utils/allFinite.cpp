#include "kernel.hpp"
#include <Kokkos_BitManipulation.hpp>
#include <cstdint>

// 0 if any conserved quantity in the interior of any block is not finite;
// python combines the ranks. Finite is read off the bits -- an exponent
// not all ones -- since a build that assumes finite arithmetic folds
// isfinite() to true, and a device build may.
PG_RANGE(cellCenters, components = ne)
struct allFinite {
  cellVecIn Q;
  static KOKKOS_INLINE_FUNCTION int finiteBits(const fpdtype x) {
#ifdef PG_SINGLE
    const uint32_t exponent = 0x7f800000u;
    return (Kokkos::bit_cast<uint32_t>(x) & exponent) != exponent;
#else
    const uint64_t exponent = 0x7ff0000000000000ull;
    return (Kokkos::bit_cast<uint64_t>(x) & exponent) != exponent;
#endif
  }
  KOKKOS_INLINE_FUNCTION void operator()(const int l, int &finite) const {
    finite = finite && finiteBits(Q(l));
  }
};

PG_ABI int pgAllFinite(const allFinite &k, const pgTiling &t) {
  int all = 1;
  reduceCellsAndComponents("check nan", t, k, Kokkos::Min<int>(all));
  return all;
}
