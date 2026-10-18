#include "kernel.hpp"
#include "thermo/doubleFlux.hpp"

// dQ is the flux difference across the cell along every axis, from the
// accumulated fluxes:
// begun here, or appended to what the chemistry source began it with
// (PG_FLUXES_APPEND, which the simulator bakes when a case has chemistry).
// Under the double flux (PG_DOUBLE_FLUX) a face holds two energy fluxes,
// F(4) in its left cell's relation and F(ne) in its right cell's, and a
// cell takes its own: it is the right cell of the face on its left and the
// left cell of the face on its right.
PG_RANGE(cellCenters, components = ne)
struct applyFlux {
  cellScalIn Jinv;
  iFaceStradVecIn iF;
  jFaceStradVecIn jF;
  kFaceStradVecIn kF;
  // read when appending
  cellVecInOut dQ;
  KOKKOS_INLINE_FUNCTION void operator()(const int l) const {
    fpdtype f;
#ifdef PG_DOUBLE_FLUX
    if (l == 4) {
      f = iF.L(ne) + jF.L(ne) + kF.L(ne) - iF.R(4) - jF.R(4) - kF.R(4);
    } else
#endif
    {
      f = iF.L(l) + jF.L(l) + kF.L(l) - iF.R(l) - jF.R(l) - kF.R(l);
    }
    f *= Jinv;
#ifdef PG_FLUXES_APPEND
    dQ(l) += f;
#else
    dQ(l) = f;
#endif
  }
};

PG_ABI void pgApplyFlux(const applyFlux &k, const pgTiling &t) {
  forCellsAndComponents("Apply current fluxes to RHS", t, k);
}
