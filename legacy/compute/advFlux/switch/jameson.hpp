// Two sensors: Jameson's on the pressure, the second difference along the
// face normal over the sum at the cell either side of the face, the larger,
// which a shock or a pressure checkerboard raises; and the jump across the
// face over the sum of gamma, rho c^2 / p from the cell's density, speed of
// sound and pressure -- the ratio of specific heats for a perfect gas, and
// for any fluid the exponent that ties its pressure to its energy -- which
// changes across a contact between fluids or a steep temperature change in
// a real gas and is what a conservative energy flux cannot carry through
// without a pressure error; a contact being a jump, its sensor is the
// first difference. Each times its gain, clipped to one; a gain of zero
// turns its sensor off. A blend takes the larger of the two; a correction
// takes the pressure sensor alone for its pressure and velocity fields,
// which a contact leaves alone and a shock does not, and the larger of the
// two for the species and the temperature, which are what a contact gets
// wrong -- so a composition gradient does not dissipate the shear that
// rides on it. The gains are the config's switchValues, pressureGain and
// gammaGain, baked in. The cells beyond the two are read: PG_STENCIL(2).
#ifndef __switchJameson_H__
#define __switchJameson_H__

#include "faces.hpp"

#if !defined(PG_JAMESON_PRESSUREGAIN) || !defined(PG_JAMESON_GAMMAGAIN)
#error "jameson takes switchValues pressureGain and gammaGain"
#endif

PG_STENCIL(2);

struct jameson : PG_RECONSTRUCT {
  using base = PG_RECONSTRUCT;
  static KOKKOS_INLINE_FUNCTION fpdtype sensor(fpdtype fm, fpdtype f,
                                               fpdtype fp) {
    return fabs(fp - 2.0 * f + fm) / fabs(fp + 2.0 * f + fm);
  }
  template <class C>
  static KOKKOS_INLINE_FUNCTION fpdtype along(const C &column, const int l) {
    return fmax(sensor(column.LL(l), column.L(l), column.R(l)),
                sensor(column.L(l), column.R(l), column.RR(l)));
  }
  template <class QC, class QP, class QH>
  static KOKKOS_INLINE_FUNCTION fpdtype gammaOf(const QC &Q, const QP &q,
                                                const QH &qh) {
    return Q(0) * qh(3) * qh(3) / q(0);
  }
  KOKKOS_INLINE_FUNCTION fpdtype shockWeight() const {
    return fmin(PG_JAMESON_PRESSUREGAIN * along(this->q, 0), 1.0);
  }
  KOKKOS_INLINE_FUNCTION fpdtype contactWeight() const {
    const fpdtype pressure = PG_JAMESON_PRESSUREGAIN * along(this->q, 0);
    const fpdtype gL = gammaOf(this->Q.L(), this->q.L(), this->qh.L()),
                  gR = gammaOf(this->Q.R(), this->q.R(), this->qh.R());
    const fpdtype gamma = PG_JAMESON_GAMMAGAIN * fabs(gR - gL) / fabs(gR + gL);
    return fmin(fmax(pressure, gamma), 1.0);
  }
  KOKKOS_INLINE_FUNCTION fpdtype weight() const { return contactWeight(); }
};

#endif
