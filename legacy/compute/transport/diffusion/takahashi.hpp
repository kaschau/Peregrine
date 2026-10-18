// Mixture-averaged diffusion from the binary coefficients, each pair's
// times Takahashi's high-pressure correction: (D p) over its dilute value
// as a function of the pair's reduced temperature and pressure, the
// critical point of the pair being its two species' weighted by their mole
// fractions renormalised over the pair. Takahashi's Table 2 gives the
// correction at seventeen reduced pressures as
// DP_R (1 - A Tr^-B) (1 - C Tr^-E), interpolated linearly in Pr between
// them, one below Pr = 0.1 and the last row's above Pr = 5; a value below
// zero, which a very low Tr gives, is floored at a tiny positive number. As
// Cantera's high-pressure transport applies it.
//
// Preparation of Generalized Chart for the Diffusion Coefficients of Gases at
// High Pressures
//     Shigeru Takahashi
//     J. Chem. Eng. Japan 1975, 7, 417-420
#ifndef __takahashi_H__
#define __takahashi_H__

#include "transport/diffusion/binary.hpp"

namespace takahashi {

constexpr int rows = 17;

// Table 2, by row: Pr, DP_R at high Tr, and A, B, C, E of the two factors
KOKKOS_INLINE_FUNCTION fpdtype table(const int column, const int row) {
  static constexpr fpdtype t[6][rows] = {
      {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5,
       3.0, 4.0, 5.0},
      {1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.02, 1.02, 1.02, 1.02, 1.03,
       1.03, 1.04, 1.05, 1.06, 1.07},
      {0.038042, 0.067433, 0.098317, 0.137610, 0.175081, 0.216376, 0.314051,
       0.385736, 0.514553, 0.599184, 0.557725, 0.593007, 0.696001, 0.790770,
       0.502100, 0.837452, 0.890390},
      {1.52267, 2.16794, 2.42910, 2.77605, 2.98256, 3.11384, 3.50264, 3.07773,
       3.54744, 3.61216, 3.41882, 3.18415, 3.37660, 3.27984, 3.39031, 3.23513,
       3.13001},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.141211, 0.278407, 0.372683,
       0.504894, 0.678469, 0.665702, 0.0, 0.602907, 0.0, 0.0},
      {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 13.45454, 14.0, 10.00900, 8.57519,
       10.37483, 11.21674, 1.0, 6.19043, 1.0, 1.0}};
  return t[column][row];
}

// one row's correction at Tr
KOKKOS_INLINE_FUNCTION fpdtype rowAt(const int row, const fpdtype Tr) {
  return table(1, row) * (1.0 - table(2, row) * pow(Tr, -table(3, row))) *
         (1.0 - table(4, row) * pow(Tr, -table(5, row)));
}

// the correction at (Pr, Tr)
KOKKOS_INLINE_FUNCTION fpdtype correction(const fpdtype Pr, const fpdtype Tr) {
  if (Pr < table(0, 0)) {
    return 1.0;
  }
  int upper = rows - 1;
  for (int j = 1; j < rows; j++) {
    if (table(0, j) > Pr) {
      upper = j;
      break;
    }
  }
  const int lower = upper == rows - 1 && table(0, upper) <= Pr ? upper : upper - 1;
  const fpdtype frac =
      lower == upper
          ? 1.0
          : (Pr - table(0, lower)) / (table(0, upper) - table(0, lower));
  const fpdtype value =
      rowAt(lower, Tr) * (1.0 - frac) + rowAt(upper, Tr) * frac;
  return fmax(value, 1.0e-20);
}

KOKKOS_INLINE_FUNCTION void coefficients(const mixtureState &s, fpdtype *D) {
  const fpdtype p = s.p, T = s.T;
  const fpdtype *X = s.X;
  binary::mixtureAveraged(s, D, [=](const int n, const int m) {
    // the pair's critical point, its species' weighted by their mole
    // fractions renormalised over the pair, each floored so a pair of
    // absent species still has one
    const fpdtype xn = fmax(X[n], 1.0e-20), xm = fmax(X[m], 1.0e-20);
    const fpdtype sum = xn + xm;
    const fpdtype Tr = T * sum / (xn * Tcrit(n) + xm * Tcrit(m));
    const fpdtype Pr = p * sum / (xn * pcrit(n) + xm * pcrit(m));
    return correction(Pr, Tr);
  });
}

} // namespace takahashi

#endif
