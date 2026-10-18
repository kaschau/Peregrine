#include "kernel.hpp"

// The sum over the range of each conserved variable's magnitude, into
// total[ne]; python combines the ranks. The scale a conservation error is
// measured against.
struct sumReducer {
  using reducer = sumReducer;
  using value_type = fpdtype;
  using result_view_type =
      Kokkos::View<fpdtype *, hostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  fpdtype &value;
  KOKKOS_INLINE_FUNCTION sumReducer(fpdtype &v) : value(v) {}
  KOKKOS_INLINE_FUNCTION void join(fpdtype &d, const fpdtype &s) const {
    d += s;
  }
  KOKKOS_INLINE_FUNCTION void init(fpdtype &v) const { v = 0.0; }
  KOKKOS_INLINE_FUNCTION fpdtype &reference() const { return value; }
  KOKKOS_INLINE_FUNCTION result_view_type view() const {
    return result_view_type(&value, 1);
  }
  KOKKOS_INLINE_FUNCTION bool references_scalar() const { return true; }
};

PG_RANGE(cellCenters)
struct totals {
  cellVecIn Q;
};
// one component's total; flat, so the shape can pin its columns
struct totalOf {
  cellVecIn Q;
  int m;
  KOKKOS_INLINE_FUNCTION void operator()(fpdtype &v) const { v += abs(Q(m)); }
};

PG_ABI void pgTotals(const totals &k, const pgTiling &t, fpdtype *total) {
  for (int m = 0; m < ne; m++) {
    fpdtype sum = 0.0;
    reduceCells("totals", t, totalOf{k.Q, m}, sumReducer(sum));
    total[m] = sum;
  }
}
