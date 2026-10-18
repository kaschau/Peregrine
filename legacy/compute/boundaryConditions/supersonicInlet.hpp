#ifndef __supersonicInlet_H__
#define __supersonicInlet_H__

#include "boundaryConditions/haloState.hpp"
#include "kernel.hpp"

namespace supersonicInlet {

// the variables the entry gives, in the order its values are baked
enum variable { p, u, v, w, T };

struct euler {
  haloInOut Q, q, qh;
  bcEntry entry;
  scalarIn tme;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const auto given = entry.at(tme());
    // the face's values, whole
    haloState(Q, q, qh, given(p), given(T), given(u), given(v), given(w), given.Y());
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

} // namespace supersonicInlet

#endif
