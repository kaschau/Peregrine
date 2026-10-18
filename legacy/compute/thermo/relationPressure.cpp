#include "kernel.hpp"
#include "thermo/doubleFlux.hpp"

// A banked state's energy taken to its relation's pressure, ahead of a
// freeze. An earlier state the integrator differences the state against is
// held in the state's variables, its energy in the relation of the step
// that banked it; through the pressure, which every relation shares,
// relationEnergy puts it in the new relation once the freeze has made it.
PG_RANGE(cellCenters)
struct relationPressure {
  cellVecInOut bank;
  cellVecIn qh;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    bank(4) = doubleFlux::pressure(bank, qh);
  }
};

PG_ABI void pgRelationPressure(const relationPressure &k, const pgTiling &t) {
  forCells("banked energy to the relation's pressure", t, k);
}
