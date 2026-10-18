#include "kernel.hpp"

// A = a*A + b*B + c*C over every element of the allocations, as copy: a
// three-array combination whose third term carries whatever step it needs
// already, a sum of stepped derivatives say. No case for a zero
// coefficient, as in axpbyStep. The arrays are ne wide.
PG_RANGE(elements, components = ne)
struct axpbypcz {
  cellVecInOut A;
  cellVecIn B, C;
  fpdtype a, b;
  fpdtype c;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    A[i] = a * A[i] + b * B[i] + c * C[i];
  }
};

PG_ABI void pgAxpbypcz(const axpbypcz &k, const pgTiling &t) {
  forElements("axpbypcz", t, k);
}
