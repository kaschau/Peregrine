// What the dual time stages share: the Weiss and Smith preconditioning's
// reference velocity.
#ifndef __dualTime_H__
#define __dualTime_H__

#include "array"
#include "kernel.hpp"
#include "math.h"
#include "vector"
#include <Kokkos_Core.hpp>

//---------------------------------------------------------------------------------------------|
//
// Dual Time with preconditioning from
//
//     Preconditioning applied to variable and constant density flows
//     Weiss, Jonathan M. and Smith, Wayne A.
//     AIAA Journal
//     1995
//     doi: 10.2514/3.12946
//
// and the unsteady floor on its reference velocity from
//
//     Dual time-stepping and preconditioning for unsteady computations
//     Venkateswaran, Sankaran and Merkle, Charles L.
//     AIAA 95-0078, 1995
//
//---------------------------------------------------------------------------------------------|

// Weiss & Smith reference velocity: the flow speed, clipped between eps*c and
// c, and no smaller than the diffusion velocity nu/dx or the unsteady
// velocity dx/(pi dt) of the smallest cell length. An inviscid call passes
// nu = 0 and a degenerate direction an infinite length, so neither binds.
static KOKKOS_INLINE_FUNCTION fpdtype
referenceVelocity(const fpdtype U, const fpdtype c, const fpdtype nu,
                  const fpdtype dI, const fpdtype dJ, const fpdtype dK,
                  const fpdtype dt) {
  const fpdtype eps = 1.0e-5;
  const fpdtype pi = 3.14159265358979323846;
  fpdtype Ur = fmax(U, eps * c);
  Ur = fmax(Ur, nu / dI);
  Ur = fmax(Ur, nu / dJ);
  Ur = fmax(Ur, nu / dK);
  Ur = fmax(Ur, fmin(fmin(dI, dJ), dK) / (pi * dt));
  return fmin(Ur, c);
}

#endif
