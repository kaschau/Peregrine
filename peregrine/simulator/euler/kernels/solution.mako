<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:macro name="fromthermo" params="Q, thermo, u">
  Q[0] = thermo.rho;
  Q[1] = thermo.rho * u[0];
  Q[2] = thermo.rho * u[1];
  Q[3] = thermo.rho * u[2];
  Q[4] = thermo.rho * (thermo.e + 0.5 * ${pg.dot("u[{i}]", i=3)});
</%pg:macro>

<%pg:macro name="efromQ" params="e, Q, rhoinv">
  e = (Q[4] - 0.5 * ${pg.dot("Q[{i}]", i=(1, 4))} * rhoinv) * rhoinv;
</%pg:macro>

<%pg:macro name="fromprims" params="Q, state, p, T, u, Y">
  MixMom mom;
  ${pg.expand("mixmom", "mom, Y")}
  EosState thermo;
  ${pg.expand("frompT", "thermo, mom, Y, p, T")}
  ${pg.expand("fromthermo", "Q, thermo, u")}
  for (int n = 0; n < ${ns} - 1; n++)
    Q[5 + n] = thermo.rho * Y(n);
  state.store(thermo);
</%pg:macro>

<%pg:macro name="massfrac" params="Y, Q, rhoinv">
  fpdtype Ylast = 1.0, Ysum = 0.0;
  for (int n = 0; n < ${ns} - 1; n++) {
    const fpdtype Yn = fmax(fmin(Q[5 + n] * rhoinv, 1.0), 0.0);
    Ylast -= Yn;
    Ysum += Yn;
  }
  const fpdtype Yscale = 1.0 / fmax(Ysum, 1.0);
  Ylast = fmax(Ylast, 0.0);
  const auto Y = [&](const int n) { return n < ${ns} - 1 ? fmax(fmin(Q[5 + n] * rhoinv, 1.0), 0.0) * Yscale : Ylast; };
</%pg:macro>
