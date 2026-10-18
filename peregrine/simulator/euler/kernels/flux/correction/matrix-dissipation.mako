<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${eos}"/>
## The Roe matrix's cross term, of the order of the flow speed, is left out; the acoustic fields are damped at the speed of sound alone
<%pg:macro name="correction" params="L, R, YL, YR, S, shock, contact, F">
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
  MixMom mom;
  ${pg.expand("mixmom", "mom, Yf")}
  fpdtype rho_p, rho_T, rho_Y[${ns}];
  ${pg.expand("drho", "rho_p, rho_T, rho_Y, mom, Yf, p, T, rho")}
  EosState face;
  ${pg.expand("frompT", "face, mom, Yf, p, T")}
  fpdtype h[${ns}];
  ${pg.expand("hspecies", "h, mom, Yf, face")}
  // sE, rho dh/dp at fixed T less one, is T rho_T / rho for any fluid
  const fpdtype cp = 0.5 * (L.cp + R.cp);
  const fpdtype H = 0.5 * (L.e + L.p / L.rho + R.e + R.p / R.rho) + ke;
  const fpdtype sE = T * rho_T * rhoinv;
  // p and u take the shock sensor alone so a contact, which gets only T and Y wrong, does not dissipate the shear riding on it
  const fpdtype acoustic = cf, convective = fabs(un);
  const fpdtype ws = shock, wc = fmax(shock, contact);
  const fpdtype dp = ws * acoustic * (R.p - L.p);
  const fpdtype dun = (R.u - L.u) * nx + (R.v - L.v) * ny + (R.w - L.w) * nz;
  const fpdtype du = ws * (convective * (R.u - L.u) + (acoustic - convective) * dun * nx);
  const fpdtype dv = ws * (convective * (R.v - L.v) + (acoustic - convective) * dun * ny);
  const fpdtype dw = ws * (convective * (R.w - L.w) + (acoustic - convective) * dun * nz);
  // dTac is the temperature's isentropic part, dT/dp at fixed entropy being -T rho_T / (rho^2 cp), damped at the acoustic speed with the pressure
  const fpdtype dTac = -T * rho_T * (R.p - L.p) / (rho * rho * cp);
  const fpdtype dT = wc * convective * (R.T - L.T) + ws * (acoustic - convective) * dTac;
  fpdtype dYs[${ns}];
  for (int n = 0; n < ${ns} - 1; n++)
    dYs[n] = wc * convective * dY[n];
  fpdtype drho = rho_p * dp + rho_T * dT;
  fpdtype dE = (rho_p * H + sE) * dp + rho * (uf * du + vf * dv + wf * dw) + (rho_T * H + rho * cp) * dT;
  for (int n = 0; n < ${ns} - 1; n++) {
    drho += rho_Y[n] * dYs[n];
    dE += (H * rho_Y[n] + rho * (h[n] - h[${ns} - 1])) * dYs[n];
  }
  const fpdtype half = -0.5 * area;
  F[0] += half * drho;
  F[1] += half * (drho * uf + rho * du);
  F[2] += half * (drho * vf + rho * dv);
  F[3] += half * (drho * wf + rho * dw);
  F[4] += half * dE;
  for (int n = 0; n < ${ns} - 1; n++)
    F[5 + n] += half * (drho * Y[n] + rho * dYs[n]);
</%pg:macro>
