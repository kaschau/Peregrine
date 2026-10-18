// The numerical diffusion the case's advective fluxes add, for the pseudo
// step to count beside the waves. A dissipative flux differences the two
// sides' states at the face by some speed, which on the grid is a diffusion
// of half that speed times the cell length; an explicit pseudo march is
// bounded by it as by any diffusion, and where the waves' bound grows with
// the root of the number of directions a diffusion's sums over them. Each
// formula says its speeds along a direction, given the normal velocity
// there, the speed of sound and the preconditioner's reference velocity and
// alpha (the speed of sound and zero without it): one for a diffusion that
// acts on every variable in every direction, which sums over the
// directions, and one for a diffusion along a single direction alone, which
// the largest direction bounds. Each formula's is advFlux/diffusion/
// <formula>.hpp's struct <formula>Diffusion; the jit forces the case's in
// ahead of a source that reaches this header and names the formulas
// PG_PRIMARY and, with a switch, PG_SECONDARY, whose diffusion counts fully
// where the switch turns it fully on.
#ifndef __dissipation_H__
#define __dissipation_H__

#include "kernel.hpp"

#ifndef PG_PRIMARY
#error "the fluxes' diffusion is the case's: -DPG_PRIMARY from the jit"
#endif
#define PG_DIFFUSION_(formula) formula##Diffusion
#define PG_DIFFUSION(formula) PG_DIFFUSION_(formula)

namespace numericalDiffusion {

// the case's fluxes' diffusion speeds along one direction, the larger of
// the primary's and the secondary's
KOKKOS_INLINE_FUNCTION void along(const fpdtype un, const fpdtype c,
                                  const fpdtype Ur, const fpdtype alpha,
                                  fpdtype &summing, fpdtype &alone) {
  PG_DIFFUSION(PG_PRIMARY)::along(un, c, Ur, alpha, summing, alone);
#ifdef PG_SECONDARY
  fpdtype summing2, alone2;
  PG_DIFFUSION(PG_SECONDARY)::along(un, c, Ur, alpha, summing2, alone2);
  summing = fmax(summing, summing2);
  alone = fmax(alone, alone2);
#endif
}

} // namespace numericalDiffusion

#endif
