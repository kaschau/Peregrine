#include "kernel.hpp"

// What dQ holds, reduced by equation over the range: the largest magnitude
// into rMax[ne], the sum of squares into rSum[ne] and the signed sum into
// rNet[ne]; python combines the ranks. Dual time reads its pseudo residual
// with it -- the right-hand side less the physical time derivative, before
// the preconditioner -- whose signed sum times 2 dt / 3 is exactly what the
// step's total of each conserved variable misses its own equation by.

// the largest magnitude, the sum of squares and the signed sum, reduced
// together
struct maxAndSum {
  fpdtype mx, sm, net;
};
struct residualReducer {
  using reducer = residualReducer;
  using value_type = maxAndSum;
  using result_view_type =
      Kokkos::View<maxAndSum *, hostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  maxAndSum &value;
  KOKKOS_INLINE_FUNCTION residualReducer(maxAndSum &v) : value(v) {}
  KOKKOS_INLINE_FUNCTION void join(maxAndSum &d, const maxAndSum &s) const {
    d.mx = fmax(d.mx, s.mx), d.sm += s.sm, d.net += s.net;
  }
  KOKKOS_INLINE_FUNCTION void init(maxAndSum &v) const {
    v = {0.0, 0.0, 0.0};
  }
  KOKKOS_INLINE_FUNCTION maxAndSum &reference() const { return value; }
  KOKKOS_INLINE_FUNCTION result_view_type view() const {
    return result_view_type(&value, 1);
  }
  KOKKOS_INLINE_FUNCTION bool references_scalar() const { return true; }
};

PG_RANGE(cellCenters)
struct residual {
  cellVecIn dQ;
};
// one component's residual; flat, so the shape can pin its columns
struct residualOf {
  cellVecIn dQ;
  int m;
  KOKKOS_INLINE_FUNCTION void operator()(maxAndSum &v) const {
    const fpdtype res = abs(dQ(m));
    v.mx = fmax(res, v.mx);
    v.sm += res * res;
    v.net += dQ(m);
  }
};

PG_ABI void pgResidual(const residual &k, const pgTiling &t, fpdtype *rMax,
                       fpdtype *rSum, fpdtype *rNet) {
  for (int m = 0; m < ne; m++) {
    maxAndSum total{0.0, 0.0, 0.0};
    reduceCells("residual", t, residualOf{k.dQ, m}, residualReducer(total));
    rMax[m] = total.mx, rSum[m] = total.sm, rNet[m] = total.net;
  }
}
