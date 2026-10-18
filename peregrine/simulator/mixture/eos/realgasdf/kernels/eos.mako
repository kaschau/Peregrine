<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.mixture.eos.realgas.kernels.eos"/>
<%include file="peregrine.simulator.mixture.eos.realgasdf.kernels.newton"/>
## the plane is rho e = theta p + eps rho + sum zeta_n rho Y_n + kappa, read off the state of the cell that froze it

<%pg:macro name="fromrhop" params="thermo, mom, Y, rho, p, Tguess">
  Cubic c;
  ${pg.expand("cubicmix", "c, mom")}
  ${pg.expand("atvolume", "c, rho")}
  const fpdtype vs = c.v + cubicsigma * c.bm, ve = c.v + cubiceps * c.bm;
  const fpdtype invvmb = 1.0 / (c.v - c.bm), invD = 1.0 / (vs * ve);
  fpdtype T = (Tguess < 1.0) ? 300.0 : Tguess;
  ${pg.expand("newtonp", "T, c, Y, p, invvmb, invD")}
  // the state at T: Z is the volume's, no root to find
  ${pg.expand("attraction", "c, Y, T")}
  fpdtype cp_ideal, h_ideal;
  ${pg.expand("ideal_cph", "cp_ideal, h_ideal, mom, T")}
  ${pg.expand("dimless", "c, p, T")}
  c.Z = p * c.invRuT * c.v;
  ${pg.expand("rootslope", "c")}
  fpdtype hdep, cpdep;
  ${pg.expand("departures", "hdep, cpdep, c, T")}
  thermo.rho = rho;
  thermo.rhoinv = 1.0 / rho;
  thermo.p = p;
  thermo.T = T;
  thermo.Z = c.Z;
  thermo.cp = cp_ideal + mom.R * cpdep;
  thermo.e = h_ideal + mom.R * T * hdep - p / rho;
  const fpdtype invZt = 1.0 / (c.Z - c.Cstar);
  ${pg.expand("gamma_c", "thermo.gamma, thermo.c, c, p, T, rho, thermo.cp, invZt")}
</%pg:macro>

<%pg:macro name="planeslopes" params="theta, epsY, mom, Y, rho, rhoinv, T, e">
  Cubic c;
  ${pg.expand("cubicmix", "c, mom")}
  ${pg.expand("atvolume", "c, rho")}
  ${pg.expand("attraction", "c, Y, T")}
  const fpdtype vs = c.v + cubicsigma * c.bm, ve = c.v + cubiceps * c.bm;
  const fpdtype invvmb = 1.0 / (c.v - c.bm), invD = 1.0 / (vs * ve), invMWmix = 1.0 / c.MWmix;
  fpdtype cp_ideal, h_ideal;
  ${pg.expand("ideal_cph", "cp_ideal, h_ideal, mom, T")}
  const fpdtype app = 0.5 * (c.S2 * c.S2 - c.dam) / T;
  const fpdtype cv = cp_ideal - mom.R + T * app * c.I * invMWmix;
  const fpdtype p = Ru * T * invvmb - c.am * invD;
  const fpdtype dpdT = Ru * invvmb - c.dam * invD;
  const fpdtype dpdrho = c.MWmix * rhoinv * rhoinv * (Ru * T * invvmb * invvmb - c.am * (vs + ve) * invD * invD);
  theta = rho * cv / dpdT;
  epsY = e - (T * dpdT - p) * rhoinv - theta * dpdrho;
</%pg:macro>

<%pg:macro name="pplane" params="p, Q, rhoinv, plane">
  fpdtype composition = 0.0;
% for n in range(ns - 1):
  composition += plane.zeta${n} * Q[5 + ${n}];
% endfor
  p = (Q[4] - 0.5 * ${pg.dot("Q[{i}]", i=(1, 4))} * rhoinv - plane.eps * Q[0] - composition - plane.kappa) / plane.theta;
</%pg:macro>

<%pg:macro name="ethermal" params="t, p, rhoinv, plane">
  t = (plane.theta * p + plane.kappa) * rhoinv;
</%pg:macro>

<%pg:macro name="ecomp" params="c0, c1, Yside, plane0, plane1">
  c0 = 0.0;
  c1 = 0.0;
% for n in range(ns - 1):
  const fpdtype y${n} = Yside(${n});
  c0 += plane0.zeta${n} * y${n};
  c1 += plane1.zeta${n} * y${n};
% endfor
</%pg:macro>

<%pg:macro name="eside" params="E0, E1, side, Yside, plane0, plane1">
  fpdtype composition[2];
  ${pg.expand("ecomp", "composition[0], composition[1], Yside, plane0, plane1")}
% for i in range(2):
  E${i} = plane${i}.theta * side.p + side.rho * (plane${i}.eps + composition[${i}] + 0.5 * ${pg.dot("side.{c}", c="uvw")}) + plane${i}.kappa;
% endfor
</%pg:macro>
