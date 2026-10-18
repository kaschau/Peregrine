<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.mixture.eos.kernels.mixing"/>
<%include file="peregrine.simulator.mixture.caloric.tdepcp.kernels.caloric"/>
<%include file="peregrine.simulator.mixture.eos.kernels.species"/>
<%include file="peregrine.simulator.mixture.eos.tpg.kernels.newton"/>
<%pg:alias name="drho" file="peregrine.simulator.mixture.eos.cpg.kernels.eos"/>

<%pg:macro name="fromrhoE" params="thermo, mom, Y, rho, e, Tguess">
  fpdtype T = (Tguess < 1.0) ? 300.0 : Tguess;
  fpdtype cp, h;
  ${pg.expand("newton", "T, cp, h, mom, e")}
  thermo.rho = rho;
  thermo.rhoinv = 1.0 / rho;
  thermo.e = e;
  thermo.T = T;
  thermo.p = rho * mom.R * T;
  thermo.cp = cp;
  thermo.gamma = cp / (cp - mom.R);
  thermo.c = sqrt(thermo.gamma * mom.R * T);
</%pg:macro>

<%pg:macro name="frompT" params="thermo, mom, Y, p, T">
  fpdtype cp, h;
  ${pg.expand("ideal_cph", "cp, h, mom, T")}
  thermo.p = p;
  thermo.T = T;
  thermo.rho = p / (mom.R * T);
  thermo.rhoinv = 1.0 / thermo.rho;
  thermo.e = h - mom.R * T;
  thermo.cp = cp;
  thermo.gamma = cp / (cp - mom.R);
  thermo.c = sqrt(thermo.gamma * mom.R * T);
</%pg:macro>
