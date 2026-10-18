<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.mixture.transport.kernels.dilute"/>
## Chung's dipole terms are left out: under the one-fluid rule a few mole percent of water would swing the dense viscosity by tens of percent, so the polar physics stays in the dilute part

## Chung, Ajlan, Lee, Starling, Ind. Eng. Chem. Res. 1988, 27, 671, in SI as Poling 5th ed. 9-5, 9-6 and 10-5 states it, the critical volume in cm^3/mol
<%pg:macro name="transport" params="mu, kappa, rho, T, lnT, X, Y, mom">
  fpdtype mudilute, kappadilute;
  ${pg.expand("dilute", "mudilute, kappadilute, T, lnT, X")}
  fpdtype sigma3 = 0.0, epssigma3 = 0.0, masssum = 0.0, omegasigma3 = 0.0, cvR = 0.0;
  for (int i = 0; i < ${ns}; i++) {
    fpdtype cpR = 0.0;
    for (int m = cppoly_terms - 1; m >= 0; m--)
      cpR = cpR * lnT + cppoly(i, m);
    cvR += X(i) * (cpR - 1.0);
    for (int j = 0; j < ${ns}; j++) {
      const fpdtype xx = X(i) * X(j);
      const fpdtype sigma2ij = chungsigma(i) * chungsigma(j);
      const fpdtype sigma3ij = sigma2ij * sqrt(sigma2ij);
      const fpdtype epsij = sqrt(chungepsk(i) * chungepsk(j));
      const fpdtype Mij = 2.0 * MW(i) * MW(j) / (MW(i) + MW(j));
      sigma3 += xx * sigma3ij;
      epssigma3 += xx * epsij * sigma3ij;
      masssum += xx * epsij * sigma2ij * sqrt(Mij);
      omegasigma3 += xx * 0.5 * (acentric(i) + acentric(j)) * sigma3ij;
    }
  }
  const fpdtype epsk = epssigma3 / sigma3;
  const fpdtype cbrtsigma3 = cbrt(sigma3);
  const fpdtype sigma2 = cbrtsigma3 * cbrtsigma3;
  const fpdtype sqrtMm = masssum / (epsk * sigma2);
  const fpdtype Mm = sqrtMm * sqrtMm;
  const fpdtype omega = omegasigma3 / sigma3;
  // its critical point, in Chung's units
  const fpdtype Tc = 1.2593 * epsk;
  const fpdtype Vc = sigma3 / (0.809 * 0.809 * 0.809);
  const fpdtype Fc = 1.0 - 0.2756 * omega;
  const fpdtype cbrtVc = cbrt(Vc);
  const fpdtype Vc23 = cbrtVc * cbrtVc;
  const fpdtype Mprime = Mm * 1.0e-3;
  fpdtype E[10], B[7];
  for (int i = 0; i < 10; i++)
    E[i] = chungvisc(i, 0) + chungvisc(i, 1) * omega;
  for (int i = 0; i < 7; i++)
    B[i] = chungcond(i, 0) + chungcond(i, 1) * omega;
  const fpdtype Tr = T / Tc;
  const fpdtype Tstar = T / epsk, invTstar = epsk / T;
  const fpdtype Omegav = 1.16145 * pow(Tstar, -0.14874) + 0.52487 * exp(-0.77320 * Tstar) + 2.16178 * exp(-2.43787 * Tstar);
  // the reduced density: rho in mol/cm^3 times Vc over six
  const fpdtype Yy = rho / Mm * 1.0e-3 * Vc / 6.0;
  const fpdtype omYy = 1.0 - Yy;
  const fpdtype G1 = (1.0 - 0.5 * Yy) / (omYy * omYy * omYy);
  // viscosity: Chung's dense eta* over its dilute eta*, the latter sqrt(T*) Fc / Omegav
  const fpdtype G2 = (E[0] * (1.0 - exp(-E[3] * Yy)) / Yy + E[1] * G1 * exp(E[4] * Yy) + E[2] * G1) / (E[0] * E[3] + E[1] + E[2]);
  const fpdtype etastarstar = E[6] * Yy * Yy * G2 * exp(E[7] + E[8] * invTstar + E[9] * invTstar * invTstar);
  const fpdtype muratio = (1.0 / G2 + E[5] * Yy) + etastarstar * Omegav / (sqrt(Tstar) * Fc);
  mu = mudilute * muratio;
  // thermal conductivity: Chung's dense lambda over its dilute, 31.2 eta0 Psi / M', eta0 in Pa s
  const fpdtype eta0 = 4.0785e-6 * sqrt(Mm * T) / (Vc23 * Omegav) * Fc;
  const fpdtype alpha = cvR - 1.5;
  const fpdtype beta = 0.7862 - 0.7109 * omega + 1.3168 * omega * omega;
  const fpdtype Z = 2.0 + 10.5 * Tr * Tr;
  const fpdtype Psi = 1.0 + alpha * ((0.215 + 0.28288 * alpha - 1.061 * beta + 0.26665 * Z) / (0.6366 + beta * Z + 1.061 * alpha * beta));
  const fpdtype lambda0 = 31.2 * eta0 * Psi / Mprime;
  const fpdtype H2 = (B[0] * (1.0 - exp(-B[3] * Yy)) / Yy + B[1] * G1 * exp(B[4] * Yy) + B[2] * G1) / (B[0] * B[3] + B[1] + B[2]);
  const fpdtype kapparatio = (1.0 / H2 + B[5] * Yy) + 3.586e-3 * sqrt(Tc / Mprime) / Vc23 * B[6] * Yy * Yy * H2 * sqrt(Tr) / lambda0;
  kappa = kappadilute * kapparatio;
</%pg:macro>
