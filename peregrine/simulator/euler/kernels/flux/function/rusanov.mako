<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:macro name="rusanov" params="L, R, YL, YR, S, F">
  const fpdtype area = sqrt(${pg.dot("S[{i}]", i=3)});
  const fpdtype unL = ${pg.dot("L.{c}", "S[{i}]", c="uvw", i=3)} / area;
  const fpdtype unR = ${pg.dot("R.{c}", "S[{i}]", c="uvw", i=3)} / area;
  const fpdtype lam = fmax(fabs(unL) + L.c, fabs(unR) + R.c) * area;
  const fpdtype UL = unL * area, UR = unR * area;
  F[0] = (0.5 * (UR * R.rho + UL * L.rho - lam * (R.rho - L.rho)));
  F[1] = (0.5 * (UR * R.rho * R.u + R.p * S[0] + UL * L.rho * L.u + L.p * S[0] - lam * (R.rho * R.u - L.rho * L.u)));
  F[2] = (0.5 * (UR * R.rho * R.v + R.p * S[1] + UL * L.rho * L.v + L.p * S[1] - lam * (R.rho * R.v - L.rho * L.v)));
  F[3] = (0.5 * (UR * R.rho * R.w + R.p * S[2] + UL * L.rho * L.w + L.p * S[2] - lam * (R.rho * R.w - L.rho * L.w)));
  const fpdtype EL = L.rho * (L.e + 0.5 * ${pg.dot("L.{c}", c="uvw")});
  const fpdtype ER = R.rho * (R.e + 0.5 * ${pg.dot("R.{c}", c="uvw")});
  F[4] = (0.5 * (UR * (ER + R.p) + UL * (EL + L.p) - lam * (ER - EL)));
  for (int n = 0; n < ${ns} - 1; n++) {
    const fpdtype rhoYL = L.rho * YL(n), rhoYR = R.rho * YR(n);
    F[5 + n] = (0.5 * (rhoYR * UR + rhoYL * UL - lam * (rhoYR - rhoYL)));
  }
</%pg:macro>
