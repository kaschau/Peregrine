#include "kernel.hpp"
#include "thermo/doubleFlux.hpp"

// The state of every cell from its conserved variables under the double
// flux: the pressure is the cell's frozen relation's, and the eos gives the
// rest at that density and pressure, T's last value its guess. The conserved
// energy stays the relation's until the next freeze; qh(4) is the eos's
// energy at the state, which the freeze resets it to. The composition is
// taken as it is; a model that needs it within bounds clips its own copy.
PG_RANGE(cellCenters)
struct stateFromRelation {
  cellVecIn Q;
  cellVecInOut q, qh;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const fpdtype rho = Q(0);
    const fpdtype rhoinv = 1.0 / rho;
    const fpdtype p = doubleFlux::pressure(Q, qh);
    const auto s =
        eos::fromRhoP(rho, p, massFractions(Q, rhoinv), q(1), keepHi(qh));
    q(0) = s.p;
    q(1) = s.T;
    qh(0) = s.gamma;
    qh(1) = s.cp;
    qh(2) = rho * s.h;
    qh(3) = s.c;
    qh(4) = rho * s.e;
  }
};

PG_ABI void pgStateFromRelation(const stateFromRelation &k,
                                const pgTiling &t) {
  forCells("state from the frozen relation", t, k);
}
