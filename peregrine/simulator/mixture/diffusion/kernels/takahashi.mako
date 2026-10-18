<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:alias name="mixavg" file="peregrine.simulator.mixture.diffusion.kernels.binary"/>
## Takahashi, J. Chem. Eng. Japan 1975, 7, 417, Table 2, as Cantera's high-pressure transport applies it
<%
table = (
    (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0),
    (1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.02, 1.02, 1.02, 1.02, 1.03, 1.03, 1.04, 1.05, 1.06, 1.07),
    (0.038042, 0.067433, 0.098317, 0.137610, 0.175081, 0.216376, 0.314051, 0.385736, 0.514553, 0.599184, 0.557725, 0.593007, 0.696001, 0.790770, 0.502100, 0.837452, 0.890390),
    (1.52267, 2.16794, 2.42910, 2.77605, 2.98256, 3.11384, 3.50264, 3.07773, 3.54744, 3.61216, 3.41882, 3.18415, 3.37660, 3.27984, 3.39031, 3.23513, 3.13001),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.141211, 0.278407, 0.372683, 0.504894, 0.678469, 0.665702, 0.0, 0.602907, 0.0, 0.0),
    (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 13.45454, 14.0, 10.00900, 8.57519, 10.37483, 11.21674, 1.0, 6.19043, 1.0, 1.0),
)
rows = len(table[0])
%>
<%pg:macro name="diffcoeffs" params="D, p, T, lnT, rhoinv, cp, kappa, MWmix, X">
  // Table 2, by row: Pr, DP_R at high Tr, and A, B, C, E of the two factors
  static constexpr fpdtype table[6][${rows}] = {
% for row in table:
      {${", ".join(repr(v) for v in row)}},
% endfor
  };
  const auto rowat = [&](const int row, const fpdtype Tr) {
    return table[1][row] * (1.0 - table[2][row] * pow(Tr, -table[3][row])) * (1.0 - table[4][row] * pow(Tr, -table[5][row]));
  };
  // one below the first pressure and the last row's past the last; the floor catches a negative value at very low Tr
  const auto correction = [&](const fpdtype Pr, const fpdtype Tr) {
    int below = 0;
% for pr in table[0]:
    below += Pr >= ${repr(pr)};
% endfor
    const int lower = below > 0 ? below - 1 : 0;
    const int upper = below < ${rows} ? below : ${rows} - 1;
    const fpdtype span = table[0][upper] - table[0][lower];
    const fpdtype frac = upper > lower ? (Pr - table[0][lower]) / span : 1.0;
    const fpdtype value = rowat(lower, Tr) * (1.0 - frac) + rowat(upper, Tr) * frac;
    return below > 0 ? fmax(value, fpdtype(1.0e-20)) : fpdtype(1.0);
  };
  // the correction's reciprocal at the pair's critical point, each mole fraction floored so a pair of absent species still has one
  const auto invfactor = [&](const int n, const int m) {
    const fpdtype xn = fmax(X(n), 1.0e-20), xm = fmax(X(m), 1.0e-20);
    const fpdtype sum = xn + xm;
    const fpdtype Tr = T * sum / (xn * Tcrit(n) + xm * Tcrit(m));
    const fpdtype Pr = p * sum / (xn * pcrit(n) + xm * pcrit(m));
    return 1.0 / correction(Pr, Tr);
  };
  ${pg.expand("mixavg", "D, p, T, lnT, MWmix, X, invfactor")}
</%pg:macro>
