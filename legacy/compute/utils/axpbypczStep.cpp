#include "kernel.hpp"

// A = a*A + b*B + c*dt*C, the three-array combination a Runge-Kutta stage
// is -- C the derivative, stepped by the step's dt read where the kernels
// run -- over every element of the allocations, as copy; no case for a zero
// coefficient, as in axpbyStep. The arrays are ne wide.
PG_RANGE(elements, components = ne)
struct axpbypczStep {
  cellVecInOut A;
  cellVecIn B, C;
  fpdtype a, b;
  fpdtype c;
  scalarIn dt;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    A[i] = a * A[i] + b * B[i] + c * dt() * C[i];
  }
};

PG_ABI void pgAxpbypczStep(const axpbypczStep &k, const pgTiling &t) {
  forElements("axpbypczStep", t, k);
}
