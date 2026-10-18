#ifndef __semSubsonicInlet_H__
#define __semSubsonicInlet_H__

#include "boundaryConditions/haloState.hpp"
#include "kernel.hpp"

namespace semSubsonicInlet {

// the variables the entry gives, in the order its values are baked
enum variable { T };

struct euler {
  haloInOut Q, q, qh;
  bcEntry entry;
  scalarIn tme;
  // the current spline interval's coefficients over the face, (component,
  // order) with the cubic first, already in grid components, and the time
  // that interval starts; the time the state is at is the solver's clock
  blockFaceIn semCoeffs;
  plainIn tBreak;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const auto given = entry.at(tme());
    const fpdtype dt = tme() - tBreak(0);
    fpdtype V[3];
    for (int v = 0; v < 3; v++) {
      V[v] = ((semCoeffs(v, 0) * dt + semCoeffs(v, 1)) * dt + semCoeffs(v, 2)) * dt +
             semCoeffs(v, 3);
    }
    // the pressure extrapolated to the halo, each layer mirrored about the
    // first interior cell; the signal's velocity, the face's temperature and
    // species
    const fpdtype p = 2.0 * q.R(0) - q.at(q.p.g + 1, 0);
    haloState(Q, q, qh, p, given(T), V[0], V[1], V[2], given.Y());
  }
};

struct postDqDxyz {
  haloInOut grads;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // neumann all gradients
    for (int l = 0; l < ne - 1; l++) {
      for (int d = 0; d < 3; d++) {
        grads.L(l, d) = grads.R(l, d);
      }
    }
  }
};

} // namespace semSubsonicInlet

#endif
