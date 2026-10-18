<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.mixture.eos.kernels.mixing"/>
<%include file="peregrine.simulator.mixture.eos.kernels.species"/>

<%pg:macro name="fromrhoE" params="thermo, mom, Y, rho, e, Tguess">
  const fpdtype cv = mom.cp - mom.R;
  thermo.rho = rho;
  thermo.rhoinv = 1.0 / rho;
  thermo.e = e;
  thermo.T = (e - mom.h298) / cv;
  thermo.p = rho * mom.R * thermo.T;
  thermo.cp = mom.cp;
  thermo.gamma = mom.cp / cv;
  thermo.c = sqrt(thermo.gamma * mom.R * thermo.T);
</%pg:macro>

<%pg:macro name="frompT" params="thermo, mom, Y, p, T">
  const fpdtype cv = mom.cp - mom.R;
  thermo.p = p;
  thermo.T = T;
  thermo.rho = p / (mom.R * T);
  thermo.rhoinv = 1.0 / thermo.rho;
  thermo.e = cv * T + mom.h298;
  thermo.cp = mom.cp;
  thermo.gamma = mom.cp / cv;
  thermo.c = sqrt(thermo.gamma * mom.R * T);
</%pg:macro>

<%pg:macro name="drho" params="rho_p, rho_T, rho_Y, mom, Y, p, T, rho">
  rho_p = rho / p;
  rho_T = -rho / T;
  const fpdtype MWmix = Ru / mom.R;
  for (int n = 0; n < ${ns} - 1; n++)
    rho_Y[n] = -rho * MWmix * (MWinv(n) - MWinv(${ns} - 1));
</%pg:macro>
