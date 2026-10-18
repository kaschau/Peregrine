<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:macro name="ausmup" params="L, R, YL, YR, S, F">
  const fpdtype area = sqrt(${pg.dot("S[{i}]", i=3)}), invarea = 1.0 / area;
  const fpdtype UL = ${pg.dot("L.{c}", "S[{i}]", c="uvw", i=3)} * invarea;
  const fpdtype UR = ${pg.dot("R.{c}", "S[{i}]", c="uvw", i=3)} * invarea;
  const fpdtype a12 = 0.5 * (R.c + L.c), inva12 = 1.0 / a12;
  const fpdtype ML = UL * inva12, MR = UR * inva12;
  const fpdtype Mbarsq = 0.5 * (UL * UL + UR * UR) * inva12 * inva12;
  const fpdtype Mco = ${values.get("cutoff-mach", 0.31622776601683794)}, Kp = 0.25, Ku = 0.75, sigma = 1.0, beta = 1.0 / 8.0;
  const fpdtype Mo = sqrt(fmin(1.0, fmax(Mbarsq, Mco * Mco)));
  const fpdtype fa = Mo * (2.0 - Mo);
  // each split is a polynomial of one side's Mach number alone, the fourth and fifth degrees on both second-degree splits of that same side
  const fpdtype M1p = 0.5 * (ML + fabs(ML)), M1m = 0.5 * (MR - fabs(MR));
  const fpdtype M2pL = 0.25 * (ML + 1.0) * (ML + 1.0), M2mL = -0.25 * (ML - 1.0) * (ML - 1.0);
  const fpdtype M2pR = 0.25 * (MR + 1.0) * (MR + 1.0), M2mR = -0.25 * (MR - 1.0) * (MR - 1.0);
  const fpdtype M4p = fabs(ML) >= 1.0 ? M1p : M2pL * (1.0 - 16.0 * beta * M2mL);
  const fpdtype M4m = fabs(MR) >= 1.0 ? M1m : M2mR * (1.0 + 16.0 * beta * M2pR);
  const fpdtype alpha = 3.0 / 16.0 * (-4.0 + 5.0 * fa * fa);
  // supersonic, the first-degree split over its Mach number is one or none
  const fpdtype p5p = fabs(ML) >= 1.0 ? (ML > 0.0 ? 1.0 : 0.0) : M2pL * ((2.0 - ML) - 16.0 * alpha * ML * M2mL);
  const fpdtype p5m = fabs(MR) >= 1.0 ? (MR < 0.0 ? 1.0 : 0.0) : M2mR * ((-2.0 - MR) + 16.0 * alpha * MR * M2pR);
  const fpdtype rho12 = 0.5 * (R.rho + L.rho);
  const fpdtype M12 = M4p + M4m - Kp * fmax(1.0 - sigma * Mbarsq, 0.0) * (R.p - L.p) * inva12 * inva12 / (fa * rho12);
  const fpdtype mdot = a12 * M12 * (M12 > 0.0 ? L.rho : R.rho) * area;
  const fpdtype p12 = p5p * L.p + p5m * R.p - Ku * p5p * p5m * (R.rho + L.rho) * fa * a12 * (UR - UL);
  const bool left = mdot > 0.0;
  const FaceState &U = left ? L : R;
  F[0] = (mdot);
  F[1] = (mdot * U.u + p12 * S[0]);
  F[2] = (mdot * U.v + p12 * S[1]);
  F[3] = (mdot * U.w + p12 * S[2]);
  F[4] = (mdot * (U.e + 0.5 * ${pg.dot("U.{c}", c="uvw")} + U.p / U.rho));
  for (int n = 0; n < ${ns} - 1; n++)
    F[5 + n] = (mdot * (left ? YL(n) : YR(n)));
</%pg:macro>
