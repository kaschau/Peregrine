#include "eos.hpp"
#include "kernel.hpp"

// The state of every cell from its conserved variables: the case's eos for
// p, T and qh; T's last value is the iterative eos's guess. The composition
// is taken as it is; a model that needs it within bounds clips its own copy.
PG_RANGE(cellCenters)
struct stateFromCons {
  cellVecIn Q;
  cellVecInOut q;
  cellVecOut qh;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const fpdtype rho = Q(0);
    const fpdtype rhoinv = 1.0 / rho;
    const fpdtype tke =
        0.5 * (Q(1) * Q(1) + Q(2) * Q(2) + Q(3) * Q(3)) * rhoinv;
    const fpdtype e = (Q(4) - tke) * rhoinv;
    const auto s =
        eos::fromCons(rho, e, massFractions(Q, rhoinv), q(1), keepHi(qh));
    q(0) = s.p;
    q(1) = s.T;
    qh(0) = s.gamma;
    qh(1) = s.cp;
    qh(2) = rho * s.h;
    qh(3) = s.c;
    qh(4) = rho * e;
  }
};

PG_ABI void pgStateFromCons(const stateFromCons &k, const pgTiling &t) {
  forCells("state from conserved", t, k);
}
