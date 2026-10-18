#ifndef __constantMassFluxSubsonicInlet_H__
#define __constantMassFluxSubsonicInlet_H__

#include "boundaryConditions/haloState.hpp"
#include "kernel.hpp"
#include "utils/normal.hpp"

namespace constantMassFluxSubsonicInlet {

// the variables the entry gives, in the order its values are baked
enum variable { mDotPerUnitArea, T };

struct euler {
  haloInOut Q, q, qh;
  bcEntry entry;
  scalarIn tme;
  blockFaceIn S;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const auto given = entry.at(tme());
    fpdtype area, nx, ny, nz;
    faceNormal(S(0), S(1), S(2), area, nx, ny, nz);

    // the target momentum: the mass flux along the face normal, into the
    // block; the pressure extrapolated to the halo, each layer mirrored
    // about the first interior cell; the face's temperature and species;
    // and, with the halo density the eos gives those, the halo velocity
    // that makes the face's average of density times velocity the target
    const fpdtype mDot = -S.outward() * given(mDotPerUnitArea);
    const fpdtype p = 2.0 * q.R(0) - q.at(q.p.g + 1, 0);
    const auto Yf = given.Y();
    const auto s = eos::fromPrims(p, given(T), Yf, keepHi(haloOf(qh)));
    const auto in = interiorOf(Q);
    const fpdtype scale = 4.0 * mDot / (Q.R(0) + s.rho);
    writeState(s, nx * scale - in.u, ny * scale - in.v, nz * scale - in.w, Yf,
               haloOf(Q), haloOf(q), haloOf(qh));
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

} // namespace constantMassFluxSubsonicInlet

#endif
