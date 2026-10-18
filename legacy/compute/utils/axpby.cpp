#include "kernel.hpp"

// A = a*A + b*B over every element of the allocations, as copy: a
// combination whose second term carries whatever step it needs already,
// a pseudo time increment say. No case for a zero coefficient, as in
// axpbyStep. The arrays are ne wide.
PG_RANGE(elements, components = ne)
struct axpby {
  cellVecInOut A;
  fpdtype a, b;
  cellVecIn B;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    A[i] = a * A[i] + b * B[i];
  }
};

PG_ABI void pgAxpby(const axpby &k, const pgTiling &t) {
  forElements("axpby", t, k);
}
