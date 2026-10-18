<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.mixture.transport.kernels.dilute"/>
<%include file="${eos}"/>
## no critical enhancement (it matters within a fifth of Tc near pc); the coefficients are fitted on SRK with kij = 0 and the residual entropy is the case's own cubic's

## Bell, PNAS 2019, 116, 4070; Yang et al., Int. J. Thermophys. 2022, 43, 183; Martinek et al., J. Chem. Eng. Data 2025, 70, 727; Li, Duan, Yang, Ind. Eng. Chem. Res. 2024, 63, 18160; Yang et al., ACS Omega 2025, 10, 6124
<%pg:macro name="transport" params="mu, kappa, rho, T, lnT, X, Y, mom">
  fpdtype mudilute, kappadilute;
  ${pg.expand("dilute", "mudilute, kappadilute, T, lnT, X")}
  ## cmu = N_A^(1/6) kB^(1/2) and ckappa = N_A^(7/6) kB^(3/2), N_A per kmol: the residuals' prefactors rhoN^(2/3) sqrt(m kB T) and rhoN^(2/3) kB sqrt(kB T / m)
  const fpdtype cmu = 1.079775473285991e-07, ckappa = 8.97775280863514e-04;
  fpdtype nmu[3] = {0.0, 0.0, 0.0}, nkappa[4] = {0.0, 0.0, 0.0, 0.0};
  fpdtype invsqrtmass = 0.0;
  for (int n = 0; n < ${ns}; n++) {
    for (int k = 0; k < 3; k++)
      nmu[k] += X(n) * resmu(n, k);
    for (int k = 0; k < 4; k++)
      nkappa[k] += X(n) * reskappa(n, k);
    invsqrtmass += Y(n) * rsqrtMW(n);
  }
  Cubic c;
  ${pg.expand("cubicmix", "c, mom")}
  ${pg.expand("attraction", "c, Y, T")}
  ## s+ = -s_res / R = -[ln(1 - b/v) + (da/dT) / (b R) ln(1 + b/v)] at the cubic's own volume, floored positive
  const fpdtype boverv = c.bm / (c.MWmix / rho + c.cm);
  const fpdtype splus = fmax(-(log(1.0 - boverv) + c.dam / (c.bm * Ru) * log(1.0 + boverv)), 1.0e-12);
  const fpdtype cbrts = cbrt(splus);
  const fpdtype invs23 = 1.0 / (cbrts * cbrts);
  const fpdtype cbrtrho = cbrt(rho);
  const fpdtype rho23 = cbrtrho * cbrtrho;
  const fpdtype cbrtinvMW = cbrt(1.0 / c.MWmix);
  const fpdtype sqrtT = sqrt(T);
  ## viscosity: ln(eta+ + 1) in s+^1.8, s+^2.4, s+^2.8, the powers from one fifth root
  const fpdtype u = pow(splus, 0.2);
  const fpdtype u3 = u * u * u;
  const fpdtype u9 = u3 * u3 * u3;
  const fpdtype u12 = u9 * u3;
  const fpdtype u14 = u12 * u * u;
  const fpdtype etaplus = exp(nmu[0] * u9 + nmu[1] * u12 + nmu[2] * u14) - 1.0;
  mu = mudilute + etaplus * invs23 * cmu * rho23 * sqrt(cbrtinvMW) * sqrtT;
  const fpdtype r = sqrt(splus);
  const fpdtype lambdaplus = nkappa[0] * splus + nkappa[1] * splus * r + nkappa[2] * splus * splus + nkappa[3] * splus * splus * r;
  kappa = kappadilute + lambdaplus * invs23 * ckappa * rho23 * cbrtinvMW * cbrtinvMW * sqrtT * invsqrtmass;
</%pg:macro>
