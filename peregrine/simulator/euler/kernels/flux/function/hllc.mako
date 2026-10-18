<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:macro name="hllc" params="L, R, YL, YR, S, F">
  const fpdtype area = sqrt(${pg.dot("S[{i}]", i=3)});
  const fpdtype nx = S[0] / area, ny = S[1] / area, nz = S[2] / area;
  const fpdtype unL = ${pg.dot("L.{c}", "S[{i}]", c="uvw", i=3)} / area;
  const fpdtype unR = ${pg.dot("R.{c}", "S[{i}]", c="uvw", i=3)} / area;
  const fpdtype SL = unL - L.c, SR = unR + R.c;
  const fpdtype Sstar = (R.p - L.p + L.rho * unL * (SL - unL) - R.rho * unR * (SR - unR)) / (L.rho * (SL - unL) - R.rho * (SR - unR));
  bool left, star;
  if (SL >= 0.0) {
    left = true, star = false;
  } else if (Sstar >= 0.0) {
    left = true, star = true;
  } else if (SR >= 0.0) {
    left = false, star = true;
  } else {
    left = false, star = false;
  }
  const FaceState &X = left ? L : R;
  const fpdtype U = left ? unL : unR, SX = left ? SL : SR;
  const fpdtype et = X.e + 0.5 * ${pg.dot("X.{c}", c="uvw")};
  const fpdtype mdot = U * X.rho * area;
  F[0] = (mdot);
  F[1] = (mdot * X.u + X.p * S[0]);
  F[2] = (mdot * X.v + X.p * S[1]);
  F[3] = (mdot * X.w + X.p * S[2]);
  F[4] = (mdot * et + U * X.p * area);
  for (int n = 0; n < ${ns} - 1; n++)
    F[5 + n] = (mdot * (left ? YL(n) : YR(n)));
  if (star) {
    // the jump is written as products with drho and dU, each an exact zero at rest, so a fused multiply-add keeps a uniform state exact
    const fpdtype rhostar = X.rho * ((SX - U) / (SX - Sstar));
    const fpdtype drho = rhostar - X.rho, dU = rhostar * (Sstar - U);
    const fpdtype jump = SX * area;
    F[0] += jump * drho;
    F[1] += jump * (drho * X.u + dU * nx);
    F[2] += jump * (drho * X.v + dU * ny);
    F[3] += jump * (drho * X.w + dU * nz);
    F[4] += jump * (drho * et + dU * (Sstar + X.p / (X.rho * (SX - U))));
    for (int n = 0; n < ${ns} - 1; n++)
      F[5 + n] += jump * drho * (left ? YL(n) : YR(n));
  }
</%pg:macro>
