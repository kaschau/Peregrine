#include "conserved.hpp"
#include "diffusion.hpp"
#include "dilute.hpp"
#include "kernel.hpp"

// Kinetic theory transport: the dilute mixture's viscosity and conductivity
// from each species' fit in ln T, mixed by the case's rule and the
// series-parallel mean; the species diffusion coefficients from the case's
// diffusion model.
PG_RANGE(cellCenters)
struct kineticTheory {
  cellVecIn Q, q, qh;
  cellVecOut qt;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const fpdtype p = q(0);
    const fpdtype T = q(1);
    const fpdtype u = log(T);

    // the mole fractions off the conserved state
    fpdtype X[ns];
    const fpdtype rhoinv = 1.0 / Q(0);
    const fpdtype MWmix = moleFractions(massFractions(Q, rhoinv), X);

    fpdtype mu, kappa;
    diluteTransport(T, u, X, mu, kappa);
    qt(0) = mu;
    qt(1) = kappa;

    fpdtype D[ns];
    diffusion::coefficients({p, T, u, rhoinv, qh(1), kappa, MWmix, X}, D);
    for (int n = 0; n <= ns - 1; n++) {
      qt(2 + n) = D[n];
    }
  }
};

PG_ABI void pgKineticTheory(const kineticTheory &k, const pgTiling &t) {
  forCells("Kinetic Theory trans props", t, k);
}
