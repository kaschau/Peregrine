#include "kernel.hpp"
#include "thermo/doubleFlux.hpp"

// Every cell's double flux relation for the step, frozen through its state,
// and its energy reset to the eos's, which the last step's relation left it
// off by: once at the start of a step, over every cell, halos too, so both
// cells at a face have theirs -- a halo's state being its neighbor's since
// the last exchange, its relation comes out the neighbor's. An integrator's
// earlier states are taken across the freeze by relationPressure and
// relationEnergy around it.
PG_RANGE(cellCenters)
struct freezeRelation {
  cellVecInOut Q, qh;
  cellVecIn q;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const fpdtype rho = Q(0), rhoinv = 1.0 / rho;
    const fpdtype &p = q(0), &T = q(1);
    Q(4) = qh(4) + doubleFlux::kinetic(Q);
    fpdtype Y[ns];
    massFractions(Q, rhoinv, Y);
    const auto Yf = [&](const int n) { return Y[n]; };
    // the slopes in pressure and density at fixed composition
    fpdtype theta, epsY;
    eos::energySlopes(rho, T, qh(4) * rhoinv, Yf, theta, epsY);
    // the slopes in the carried species at fixed density and pressure, the
    // last species giving way: the partial enthalpy difference, less cp
    // times the temperature change that holds the density
    fpdtype rho_p, rho_T, rho_Y[ns];
    eos::densityDerivatives(p, T, rho, Yf, rho_p, rho_T, rho_Y);
    const auto hi = eos::enthalpies(T, qh);
    fpdtype zetaY = 0.0;
    for (int n = 0; n < ns - 1; n++) {
      const fpdtype zeta = hi(n) - hi(ns - 1) - qh(1) * rho_Y[n] / rho_T;
      qh(doubleFlux::slotZeta + n) = zeta;
      zetaY += zeta * Y[n];
    }
    qh(doubleFlux::slotTheta) = theta;
    // the density slope at fixed partial densities of the carried species
    qh(doubleFlux::slotEps) = epsY - zetaY;
    qh(doubleFlux::slotKappa) = qh(4) - epsY * rho - theta * p;
  }
};

PG_ABI void pgFreezeRelation(const freezeRelation &k, const pgTiling &t) {
  forCells("freeze the double flux relation", t, k);
}
