#include "kernel.hpp"
#include "thermo/doubleFlux.hpp"

// A banked state's pressure, which relationPressure left in its energy
// slot, taken to the energy in the relation just frozen: the state's
// earlier states then read in the same variables as the state does for
// the step.
PG_RANGE(cellCenters)
struct relationEnergy {
  cellVecInOut bank;
  cellVecIn qh;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    bank(4) = doubleFlux::energy(bank(4), bank, qh);
  }
};

PG_ABI void pgRelationEnergy(const relationEnergy &k, const pgTiling &t) {
  forCells("banked pressure to the relation's energy", t, k);
}
