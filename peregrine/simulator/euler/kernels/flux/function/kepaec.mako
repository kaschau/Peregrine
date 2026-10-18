<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:macro name="kepaec" params="L, R, YL, YR, S, F">
  const fpdtype uf = 0.5 * (R.u + L.u), vf = 0.5 * (R.v + L.v), wf = 0.5 * (R.w + L.w);
  const fpdtype U = ${pg.dot("S[{i}]", "{c}f", i=3, c="uvw")};
  const fpdtype pf = 0.5 * (R.p + L.p);
  const fpdtype rho = 0.5 * (R.rho + L.rho);
  const fpdtype C = rho * U;
  F[0] = (C);
  F[1] = (C * uf + pf * S[0]);
  F[2] = (C * vf + pf * S[1]);
  F[3] = (C * wf + pf * S[2]);
  ## the harmonic mean of cv T is the entropy-consistent thermal flux; on the whole e it is singular where e changes sign, so the rest of e is averaged arithmetically
  const fpdtype Kj = C * 0.5 * ${pg.dot("R.{c}", "L.{c}", c="uvw")};
  const fpdtype Pj = 0.5 * (L.p * ${pg.dot("R.{c}", "S[{i}]", c="uvw", i=3)} + R.p * ${pg.dot("L.{c}", "S[{i}]", c="uvw", i=3)});
  const fpdtype thL = L.cp / L.gamma * L.T, thR = R.cp / R.gamma * R.T;
  const fpdtype ef = 2.0 * thL * thR / (thL + thR) + 0.5 * ((L.e - thL) + (R.e - thR));
  F[4] = (ef * C + Kj + Pj);
  for (int n = 0; n < ${ns} - 1; n++)
    F[5 + n] = (0.5 * (YR(n) + YL(n)) * C);
</%pg:macro>
