// The numerical diffusion the matrix dissipation adds: its acoustic fields
// at c' in every direction, its convective ones at |u'|; the largest wave
// speed on everything bounds it, rusanov's.
#ifndef __diffusionMatrixDissipation_H__
#define __diffusionMatrixDissipation_H__

#include "advFlux/diffusion/rusanov.hpp"

struct matrixDissipationDiffusion : rusanovDiffusion {};

#endif
