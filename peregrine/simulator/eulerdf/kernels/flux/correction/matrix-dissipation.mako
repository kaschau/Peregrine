<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${eos}"/>
## the density derivatives, rho / p and -rho / T and none by the species, are the same in both planes, so only the energy row is per plane
<%pg:macro name="correction" params="L, R, YL, YR, S, shock, contact, F"
           state="in at L, R">
  const fpdtype area = sqrt(${pg.dot("S[{i}]", i=3)}), areainv = 1.0 / area;
  const fpdtype nx = S[0] * areainv, ny = S[1] * areainv, nz = S[2] * areainv;
  const fpdtype rho = 0.5 * (L.rho + R.rho), rhoinv = 1.0 / rho;
  const fpdtype p = 0.5 * (L.p + R.p), T = 0.5 * (L.T + R.T), cf = 0.5 * (L.c + R.c);
  const fpdtype uf = 0.5 * (L.u + R.u), vf = 0.5 * (L.v + R.v), wf = 0.5 * (L.w + R.w);
  const fpdtype un = uf * nx + vf * ny + wf * nz;
  const fpdtype ke = 0.5 * (uf * uf + vf * vf + wf * wf);
  fpdtype Y[${ns}], dY[${ns}];
  for (int n = 0; n < ${ns} - 1; n++) {
    Y[n] = 0.5 * (YL(n) + YR(n));
    dY[n] = YR(n) - YL(n);
  }
  const auto Yf = [&](const int n) { return Y[n]; };
  const fpdtype rho_p = rho / p, rho_T = -rho / T;
  const fpdtype acoustic = cf, convective = fabs(un);
  const fpdtype ws = shock, wc = fmax(shock, contact);
  const fpdtype dp = ws * acoustic * (R.p - L.p);
  const fpdtype dun = (R.u - L.u) * nx + (R.v - L.v) * ny + (R.w - L.w) * nz;
  const fpdtype du = ws * (convective * (R.u - L.u) + (acoustic - convective) * dun * nx);
  const fpdtype dv = ws * (convective * (R.v - L.v) + (acoustic - convective) * dun * ny);
  const fpdtype dw = ws * (convective * (R.w - L.w) + (acoustic - convective) * dun * nz);
  fpdtype dYs[${ns}];
  for (int n = 0; n < ${ns} - 1; n++)
    dYs[n] = wc * convective * dY[n];
  // the temperature jump damped at a plane's cp: its isentropic part, dT/dp at fixed entropy being -T rho_T / (rho^2 cp), goes with the pressure
  const auto damped = [&](const fpdtype cp) {
    const fpdtype dTac = -T * rho_T * (R.p - L.p) / (rho * rho * cp);
    return wc * convective * (R.T - L.T) + ws * (acoustic - convective) * dTac;
  };
  const fpdtype cpL = ((stateL.theta + 1.0) * p + stateL.kappa) / (rho * T);
  const fpdtype cpR = ((stateR.theta + 1.0) * p + stateR.kappa) / (rho * T);
  const fpdtype dT = damped(0.5 * (cpL + cpR));
  const fpdtype drho = rho_p * dp + rho_T * dT;
  const fpdtype half = -0.5 * area;
  F[0] += half * drho;
  F[1] += half * (drho * uf + rho * du);
  F[2] += half * (drho * vf + rho * dv);
  F[3] += half * (drho * wf + rho * dw);
  for (int n = 0; n < ${ns} - 1; n++)
    F[5 + n] += half * (drho * Y[n] + rho * dYs[n]);
  fpdtype composition[2];
  ${pg.expand("ecomp", "composition[0], composition[1], Yf, stateL, stateR")}
% for i, (row, plane, cp) in enumerate(((4, "stateL", "cpL"), (ne, "stateR", "cpR"))):
  {
    const fpdtype thermal = (${plane}.theta + 1.0) * p + ${plane}.kappa;
    const fpdtype H = ${plane}.eps + composition[${i}] + ke + thermal * rhoinv;
    const fpdtype sE = -1.0 - ${plane}.kappa / p;
    const fpdtype dTp = damped(${cp});
    fpdtype dE = (rho_p * H + sE) * dp + rho * (uf * du + vf * dv + wf * dw) + (rho_T * H + rho * ${cp}) * dTp;
% for n in range(ns - 1):
    dE += rho * ${plane}.zeta${n} * dYs[${n}];
% endfor
    F[${row}] += half * dE;
  }
% endfor
</%pg:macro>
