<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.mixture.eos.kernels.mixing"/>
<%include file="peregrine.simulator.mixture.caloric.tdepcp.kernels.caloric"/>
<%include file="peregrine.simulator.mixture.eos.realgas.kernels.newton"/>
## the modulus keeps sqrt(a_i) positive past Tr = (1 + 1/m)^2, where Soave's bracket changes sign; opened as |(1 + m_i) - sqrt(T) m_i / sqrt(Tc_i)|, one fma on two baked constants
<%def name="modulus(n)">fabs(fma(-sqrtT, cubicmTc(${n}), cubiconepm(${n})))</%def>
## sqrt(a_i) whole, or per unit of Y_i / MW_i for the mixing sums; both need sqrtT in scope
<%def name="sqrta(n)">cubicsqrtac(${n}) * ${modulus(n)}</%def>
<%def name="sqrtaMW(n)">cubicsqrtacMW(${n}) * ${modulus(n)}</%def>

## cubicmix, attraction, zsolve or dimless, then departures fill a Cubic in turn; ratios and hpartial read what they left
<%pg:macro name="cubicmix" params="c, mom">
  c.MWmix = Ru / mom.R;
  c.bm = mom.b * c.MWmix;
  c.cm = mom.c * c.MWmix;
  c.S2 = mom.da * c.MWmix;
</%pg:macro>

## kij = 0: the pair sums collapse to am = S1^2 and dam/dT = -S1 S2 / sqrt(T)
<%pg:macro name="attraction" params="c, Y, T">
  const fpdtype sqrtT = sqrt(T);
  fpdtype s1 = 0.0, Ylast = 1.0;
  for (int n = 0; n < ${ns} - 1; n++) {
    const fpdtype Yn = Y(n);
    Ylast -= Yn;
    s1 = fma(Yn, ${sqrtaMW("n")}, s1);
  }
  s1 = fma(Ylast, ${sqrtaMW(f"{ns} - 1")}, s1);
  c.S1 = s1 * c.MWmix;
  c.am = c.S1 * c.S1;
  c.dam = -c.S1 * c.S2 / sqrtT;
</%pg:macro>

<%pg:macro name="dimless" params="c, p, T">
  c.invRuT = 1.0 / (Ru * T);
  const fpdtype pRT = p * c.invRuT;
  c.Astar = c.am * pRT * c.invRuT;
  c.Bstar = c.bm * pRT;
  c.Cstar = c.cm * pRT;
  const fpdtype B = c.Bstar, B2 = B * B;
  c.z1 = c.Astar + cubicw * B2 - cubicu * B - cubicu * B2;
  c.z2 = -(1.0 + B - cubicu * B);
</%pg:macro>

## Cardano; a root at or below B* is no physical volume, the stable phase is the least Gibbs energy of the rest, the largest when none remains
<%pg:macro name="zsolve" params="c, p, T">
  ${pg.expand("dimless", "c, p, T")}
  const fpdtype B = c.Bstar, B2 = B * B, z1 = c.z1, z2 = c.z2;
  const fpdtype z0 = -(c.Astar * B + cubicw * B2 * (1.0 + B));
  const fpdtype Q = (z2 * z2 - 3.0 * z1) * (1.0 / 9.0);
  const fpdtype R = (2.0 * z2 * z2 * z2 - 9.0 * z2 * z1 + 27.0 * z0) * (1.0 / 54.0);
  const fpdtype M = R * R - Q * Q * Q;
  const fpdtype z2o3 = z2 * (1.0 / 3.0);
  if (M > 0.0) {
    // one real root
    const fpdtype S = -copysign(cbrt(fabs(R) + sqrt(M)), R);
    c.Z = S + Q / S - z2o3;
  } else {
    // three, by the trigonometric form
    const fpdtype sqQ = sqrt(Q), twopio3 = 2.0943951023931954923;
    const fpdtype theta3 = acos(R / (Q * sqQ)) * (1.0 / 3.0);
    const fpdtype roots[3] = {-2.0 * sqQ * cos(theta3) - z2o3,
                              -2.0 * sqQ * cos(theta3 + twopio3) - z2o3,
                              -2.0 * sqQ * cos(theta3 - twopio3) - z2o3};
    // the roots above the covolume are the candidates and the largest is one whenever any is; a lone candidate, the hot gas's case, needs no Gibbs energy formed
    fpdtype Z = fmax(roots[0], fmax(roots[1], roots[2]));
    int above = 0;
    for (int k = 0; k < 3; k++)
      above += roots[k] > B;
    if (above > 1) {
      // several: the one of least Gibbs energy departure, g / (Ru T), is the stable phase's
      const fpdtype AoBse = c.Astar / (B * (cubicsigma - cubiceps));
      fpdtype gmin = 0.0;
      bool found = false;
      for (int k = 0; k < 3; k++) {
        const fpdtype z = roots[k];
        if (z <= B)
          continue;
        const fpdtype g = z - 1.0 - log(z - B) + AoBse * log((z + cubiceps * B) / (z + cubicsigma * B));
        if (!found || g < gmin) {
          gmin = g;
          Z = z;
          found = true;
        }
      }
    }
    c.Z = Z;
  }
  ${pg.expand("rootslope", "c")}
</%pg:macro>

## 1 over the cubic's slope at its root, which both of Z's derivatives divide by
<%pg:macro name="rootslope" params="c">
  c.invslope = 1.0 / (3.0 * c.Z * c.Z + 2.0 * c.Z * c.z2 + c.z1);
</%pg:macro>

<%pg:macro name="dZ" params="out, c, dAstar, dBstar">
  const fpdtype B = c.Bstar, Z = c.Z, Z2 = Z * Z;
  const fpdtype dz0 = -(B * dAstar + (c.Astar + (2.0 * B + 3.0 * B * B) * cubicw) * dBstar);
  const fpdtype dz1 = dAstar + (2.0 * B * (cubicw - cubicu) - cubicu) * dBstar;
  const fpdtype dz2 = -(1.0 - cubicu) * dBstar;
  out = -(dz2 * Z2 + dz1 * Z + dz0) * c.invslope;
</%pg:macro>

## the volume translation moves hdep by -C* and leaves cpdep alone; q = a / (b Ru T), qd = (da/dT) / (b Ru) and the log ratio stay on the point for hpartial
<%pg:macro name="departures" params="hdep, cpdep, c, T">
  const fpdtype Tinv = Ru * c.invRuT;
  const fpdtype dlnamdT = c.dam / c.am;
  ${pg.expand("dZ", "c.dZdT, c, -2.0 * c.Astar * Tinv * (1.0 - 0.5 * T * dlnamdT), -c.Bstar * Tinv")}
  const fpdtype x = c.Z / c.Bstar, xm1 = x - 1.0;
  const fpdtype D = x * x + cubicu * x + cubicw;
  c.logratio = log((x + cubiceps) / (x + cubicsigma));
  const fpdtype invBmRu = 1.0 / (c.bm * Ru);
  const fpdtype hres = invBmRu / (cubicsigma - cubiceps) * (c.am * Tinv - c.dam) * c.logratio;
  c.qd = c.dam * invBmRu;
  c.q = c.am * invBmRu * Tinv;
  const fpdtype num = D - c.qd * xm1;
  cpdep = hres * 0.5 * T * dlnamdT + num * num / (D * D - c.q * (2.0 * x + cubicu) * xm1 * xm1) - 1.0;
  hdep = hres + (c.Z - 1.0) - c.Cstar;
</%pg:macro>

<%pg:macro name="gamma_c" params="gamma, cs, c, p, T, rho, cp, invZt">
  const fpdtype invp = 1.0 / p, Tinv = Ru * c.invRuT;
  const fpdtype dZtdT = c.dZdT + c.Cstar * Tinv;
  fpdtype dZdp;
  ${pg.expand("dZ", "dZdp, c, c.Astar * invp, c.Bstar * invp")}
  const fpdtype dZtdp = dZdp - c.Cstar * invp;
  const fpdtype drhodp = rho * invp * (1.0 - p * dZtdp * invZt);
  // drho/dT over rho
  const fpdtype r = -Tinv * (1.0 + T * dZtdT * invZt);
  // gamma over drho/dp is 1 over the bracket, which is c^2
  const fpdtype csq = 1.0 / (drhodp - (T / cp) * r * r);
  gamma = drhodp * csq;
  cs = sqrt(fabs(csq));
</%pg:macro>

<%pg:macro name="frompT" params="thermo, mom, Y, p, T">
  Cubic c;
  ${pg.expand("cubicmix", "c, mom")}
  ${pg.expand("attraction", "c, Y, T")}
  fpdtype cp_ideal, h_ideal;
  ${pg.expand("ideal_cph", "cp_ideal, h_ideal, mom, T")}
  ${pg.expand("zsolve", "c, p, T")}
  fpdtype hdep, cpdep;
  ${pg.expand("departures", "hdep, cpdep, c, T")}
  thermo.p = p;
  thermo.T = T;
  thermo.Z = c.Z;
  thermo.cp = cp_ideal + mom.R * cpdep;
  const fpdtype Zt = c.Z - c.Cstar, invZt = 1.0 / Zt, porho = Zt * mom.R * T;
  thermo.rho = p * c.invRuT * c.MWmix * invZt;
  thermo.rhoinv = 1.0 / thermo.rho;
  thermo.e = h_ideal + mom.R * T * hdep - porho;
  ${pg.expand("gamma_c", "thermo.gamma, thermo.c, c, p, T, thermo.rho, thermo.cp, invZt")}
</%pg:macro>

<%pg:macro name="atvolume" params="c, rho">
  c.v = c.MWmix / rho + c.cm;
  c.I = log((c.v + cubicsigma * c.bm) / (c.v + cubiceps * c.bm)) / ((cubicsigma - cubiceps) * c.bm);
</%pg:macro>

<%pg:macro name="fromrhoE" params="thermo, mom, Y, rho, e, Tguess">
  Cubic c;
  ${pg.expand("cubicmix", "c, mom")}
  ${pg.expand("atvolume", "c, rho")}
  const fpdtype invMWmix = 1.0 / c.MWmix;
  fpdtype T = (Tguess < 1.0) ? 300.0 : Tguess;
  fpdtype cp_ideal, h_ideal;
  ${pg.expand("newton", "T, cp_ideal, h_ideal, c, Y, mom, e, invMWmix")}
  // the state at T: p from the volume, and Z is the volume's, no root to find
  ${pg.expand("attraction", "c, Y, T")}
  ${pg.expand("ideal_cph", "cp_ideal, h_ideal, mom, T")}
  const fpdtype vs = c.v + cubicsigma * c.bm, ve = c.v + cubiceps * c.bm;
  const fpdtype p = Ru * T / (c.v - c.bm) - c.am / (vs * ve);
  ${pg.expand("dimless", "c, p, T")}
  c.Z = p * c.invRuT * c.v;
  ${pg.expand("rootslope", "c")}
  fpdtype hdep, cpdep;
  ${pg.expand("departures", "hdep, cpdep, c, T")}
  thermo.rho = rho;
  thermo.rhoinv = 1.0 / rho;
  thermo.e = e;
  thermo.p = p;
  thermo.T = T;
  thermo.Z = c.Z;
  thermo.cp = cp_ideal + mom.R * cpdep;
  const fpdtype invZt = 1.0 / (c.Z - c.Cstar);
  ${pg.expand("gamma_c", "thermo.gamma, thermo.c, c, p, T, rho, thermo.cp, invZt")}
</%pg:macro>

## what every species' partial enthalpy shares at a cubic's point, once: ln T, 1 / T, sqrt T, I = ln((Z + sigma B*) / (Z + eps B*)) / (sigma - eps) off the point's log ratio and its T derivative, d ln(Z - B*) / dT, 1 / S1, 1 / bm and 1 / (S1^2 sqrt T), which scales every species' rT
<%pg:macro name="hpartialpoint" params="hp, c, T">
  hp.lnT = log(T);
  hp.Tinv = Ru * c.invRuT;
  hp.sqrtT = sqrt(T);
  const fpdtype Z = c.Z, B = c.Bstar, ZT = c.dZdT, BT = -B * hp.Tinv;
  const fpdtype invZsB = 1.0 / (Z + cubicsigma * B), invZeB = 1.0 / (Z + cubiceps * B);
  hp.Iz = -c.logratio / (cubicsigma - cubiceps);
  hp.IzT = ((ZT + cubicsigma * BT) * invZsB - (ZT + cubiceps * BT) * invZeB) / (cubicsigma - cubiceps);
  hp.dlnZmB = (ZT - BT) / (Z - B);
  hp.invS1 = 1.0 / c.S1;
  hp.invBm = 1.0 / c.bm;
  hp.invamsqrtT = hp.invS1 * hp.invS1 / hp.sqrtT;
</%pg:macro>

## -Ru T^2 d ln phi_n / dT with ln phi_n = (b_n/b)(Z - 1) - ln(Z - B*) - qbar_n I, qbar_n = q (2 sqrt(a_n) / S1 - b_n/b), q = a / (b Ru T), kij = 0; plus the ideal part and the translation's -p c_n
<%pg:macro name="hpartial" params="h, n, hp, c, p, T">
  fpdtype hRT = 0.0;
  for (int m = hpoly_terms - 1; m >= 0; m--)
    hRT = hRT * hp.lnT + hpoly(n, m);
  hRT += h298(n) * hp.Tinv;
  // q's temperature derivative off the point's two q's
  const fpdtype qT = (c.qd - c.q) * hp.Tinv;
  // 2 sqrt(a_n) / S1 and its temperature derivative
  const fpdtype sqrtT = hp.sqrtT;
  const fpdtype sqrtan = ${sqrta("n")};
  const fpdtype r = 2.0 * sqrtan * hp.invS1;
  const fpdtype rT = (sqrtan * c.S2 - cubicda(n) * c.S1) * hp.invamsqrtT;
  const fpdtype bb = cubicb(n) * hp.invBm;
  const fpdtype qbar = c.q * (r - bb), qbarT = qT * (r - bb) + c.q * rT;
  const fpdtype dlnphidT = bb * c.dZdT - hp.dlnZmB - qbarT * hp.Iz - qbar * hp.IzT;
  h = (hRT * T * Ru - Ru * T * T * dlnphidT - vshift(n) * p) * MWinv(n);
</%pg:macro>

## each mixing sum moves by itself times dlnMWmix plus MWmix (f_n / MW_n - f_last / MW_last), the last species giving way; am = S1^2, so dam = 2 S1 dS1
<%pg:macro name="drho" params="rho_p, rho_T, rho_Y, mom, Y, p, T, rho">
  Cubic c;
  ${pg.expand("cubicmix", "c, mom")}
  ${pg.expand("attraction", "c, Y, T")}
  ${pg.expand("zsolve", "c, p, T")}
  fpdtype hdep, cpdep;
  ${pg.expand("departures", "hdep, cpdep, c, T")}
  const fpdtype invp = 1.0 / p, Tinv = Ru * c.invRuT, sqrtT = sqrt(T);
  const fpdtype Zt = c.Z - c.Cstar, dZtdT = c.dZdT + c.Cstar * Tinv, invZt = 1.0 / Zt;
  fpdtype dZdp;
  ${pg.expand("dZ", "dZdp, c, c.Astar * invp, c.Bstar * invp")}
  rho_p = rho * invp * (1.0 - (p * dZdp - c.Cstar) * invZt);
  rho_T = -rho * Tinv * (1.0 + T * dZtdT * invZt);
  const fpdtype twoS1 = 2.0 * c.S1;
  const fpdtype invRuT = Tinv / Ru, pRT = p * invRuT;
  const fpdtype MWinv_last = MWinv(${ns} - 1);
  const fpdtype sqrta_last = ${sqrtaMW(f"{ns} - 1")};
  const fpdtype b_last = cubicbMW(${ns} - 1), c_last = vshiftMW(${ns} - 1);
  for (int n = 0; n < ${ns} - 1; n++) {
    const fpdtype dlnMWmix = -c.MWmix * (MWinv(n) - MWinv_last);
    const fpdtype dam = twoS1 * (c.S1 * dlnMWmix + c.MWmix * (${sqrtaMW("n")} - sqrta_last));
    const fpdtype dbm = c.bm * dlnMWmix + c.MWmix * (cubicbMW(n) - b_last);
    const fpdtype dcm = c.cm * dlnMWmix + c.MWmix * (vshiftMW(n) - c_last);
    fpdtype dZdY;
    ${pg.expand("dZ", "dZdY, c, dam * pRT * invRuT, dbm * pRT")}
    rho_Y[n] = rho * (dlnMWmix - (dZdY - dcm * pRT) * invZt);
  }
</%pg:macro>

<%pg:macro name="hspecies" params="h, mom, Y, state">
  const fpdtype p = state.p, T = state.T;
  Cubic c;
  ${pg.expand("cubicmix", "c, mom")}
  ${pg.expand("attraction", "c, Y, T")}
  ${pg.expand("dimless", "c, p, T")}
  // the root the state holds, no cubic to solve
  c.Z = state.Z;
  ${pg.expand("rootslope", "c")}
  fpdtype hdep, cpdep;
  ${pg.expand("departures", "hdep, cpdep, c, T")}
  Hpoint hp;
  ${pg.expand("hpartialpoint", "hp, c, T")}
  for (int n = 0; n < ${ns}; n++) {
    fpdtype hn;
    ${pg.expand("hpartial", "hn, n, hp, c, p, T")}
    h[n] = hn;
  }
</%pg:macro>
