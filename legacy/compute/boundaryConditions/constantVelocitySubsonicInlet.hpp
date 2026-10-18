#ifndef __constantVelocitySubsonicInlet_H__
#define __constantVelocitySubsonicInlet_H__

#include "boundaryConditions/haloState.hpp"
#include "kernel.hpp"

namespace constantVelocitySubsonicInlet {

// the variables the entry gives, in the order its values are baked
enum variable { u, v, w, T };

struct euler {
  haloInOut Q, q, qh;
  bcEntry entry;
  scalarIn tme;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const auto given = entry.at(tme());
    // the pressure extrapolated to the halo, each layer mirrored about the
    // first interior cell; the face's velocity, temperature and species
    const fpdtype p = 2.0 * q.R(0) - q.at(q.p.g + 1, 0);
    haloState(Q, q, qh, p, given(T), given(u), given(v), given(w), given.Y());
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

} // namespace constantVelocitySubsonicInlet

#endif
