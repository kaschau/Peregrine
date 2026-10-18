// Second order MUSCL: each side's state at the face is its cell's, moved
// half a cell along a limited slope. The primitives are reconstructed --
// pressure, temperature, velocity and mass fractions, each on its own
// limiter -- and the equation of state gives the rest of the face state
// from them, so a face state is a state of the eos whatever the limiters
// did. The limiter is PG_LIMITER, forced in ahead by the jit, in Sweby's
// form: phi of the ratio of the cell's upwind difference, away from the
// face, to the one across the face, times the one across the face, on
// both sides alike, so an upwind-biased limiter biases each side upwind.
#ifndef __reconstructMuscl_H__
#define __reconstructMuscl_H__

#include "advFlux/faceState.hpp"
#include "faces.hpp"
#include "thermo/eos.hpp"

PG_STENCIL(2);

struct muscl {
  cellStradVecIn Q, q, qh, dIJK;
  faceVecOut F;
  faceVecIn A;
  scalarIn dt;

  // the limiter's phi of the slope ratio, guarding the flat side: a slope
  // against none is the limiter's limit, none against none is none
  static KOKKOS_INLINE_FUNCTION fpdtype phi(fpdtype dm, fpdtype dp) {
    if (dp == 0.0)
      return dm == 0.0 ? 0.0 : PG_LIMITER::limit;
    return PG_LIMITER::phi(dm / dp);
  }
  // one variable at the four cells: the limiter either side of the face,
  // each of its cell's upwind difference against the one across the face,
  // and the face values they give
  struct slope {
    fpdtype phiL, phiR;
    KOKKOS_INLINE_FUNCTION slope(fpdtype vLL, fpdtype vL, fpdtype vR,
                                 fpdtype vRR)
        : phiL(phi(vL - vLL, vR - vL)), phiR(phi(vRR - vR, vR - vL)) {}
    KOKKOS_INLINE_FUNCTION fpdtype left(fpdtype vL, fpdtype vR) const {
      return vL + 0.5 * phiL * (vR - vL);
    }
    KOKKOS_INLINE_FUNCTION fpdtype right(fpdtype vL, fpdtype vR) const {
      return vR - 0.5 * phiR * (vR - vL);
    }
  };
  // one variable at the four cells, and its two face values
  struct faced {
    fpdtype L, R;
    KOKKOS_INLINE_FUNCTION faced(fpdtype vLL, fpdtype vL, fpdtype vR,
                                 fpdtype vRR) {
      const slope s(vLL, vL, vR, vRR);
      L = s.left(vL, vR);
      R = s.right(vL, vR);
    }
  };
  struct sides {
    faceState L, R;
    // the mass fractions at the face, which the species fluxes take with
    // the face's density
    fpdtype YL[ns - 1], YR[ns - 1];
  };

  // one side's state from its face primitives through the eos
  static KOKKOS_INLINE_FUNCTION void
  state(const fpdtype p, const fpdtype T, const fpdtype *u, const fpdtype *Y,
        faceState &X) {
    fpdtype last = 1.0;
    for (int n = 0; n < ns - 1; n++) {
      last -= Y[n];
    }
    const auto Yof = [=](const int n) { return n < ns - 1 ? Y[n] : last; };
    const auto s = eos::fromPrims(p, T, Yof, [](const int, const fpdtype) {});
    X.rho = s.rho;
    X.p = p;
    X.c = s.c;
    X.rhou = s.rho * u[0], X.rhov = s.rho * u[1], X.rhow = s.rho * u[2];
    X.E = s.rho * (s.e + 0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]));
  }

  KOKKOS_INLINE_FUNCTION sides states() const {
    sides s;
    const fpdtype rhoinvLL = 1.0 / Q.LL(0), rhoinvL = 1.0 / Q.L(0),
                  rhoinvR = 1.0 / Q.R(0), rhoinvRR = 1.0 / Q.RR(0);
    const faced p(q.LL(0), q.L(0), q.R(0), q.RR(0));
    const faced T(q.LL(1), q.L(1), q.R(1), q.RR(1));
    fpdtype uL[3], uR[3];
    for (int d = 0; d < 3; d++) {
      const faced u(Q.LL(1 + d) * rhoinvLL, Q.L(1 + d) * rhoinvL,
                    Q.R(1 + d) * rhoinvR, Q.RR(1 + d) * rhoinvRR);
      uL[d] = u.L, uR[d] = u.R;
    }
    for (int n = 0; n < ns - 1; n++) {
      const faced Y(Q.LL(5 + n) * rhoinvLL, Q.L(5 + n) * rhoinvL,
                    Q.R(5 + n) * rhoinvR, Q.RR(5 + n) * rhoinvRR);
      s.YL[n] = Y.L, s.YR[n] = Y.R;
    }
    state(p.L, T.L, uL, s.YL, s.L);
    state(p.R, T.R, uR, s.YR, s.R);
    return s;
  }
  KOKKOS_INLINE_FUNCTION void species(const sides &s, int n, fpdtype &L,
                                      fpdtype &R) const {
    L = s.L.rho * s.YL[n];
    R = s.R.rho * s.YR[n];
  }
};

#endif
