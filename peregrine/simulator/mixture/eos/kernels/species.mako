<%namespace name="pg" module="peregrine.codegen.makoutil"/>

<%pg:macro name="cpRspecies" params="cpR, n, u">
  cpR = 0.0;
  for (int m = cppoly_terms - 1; m >= 0; m--)
    cpR = cpR * u + cppoly(n, m);
</%pg:macro>

<%pg:macro name="hRTspecies" params="hRT, n, u, Tinv">
  hRT = 0.0;
  for (int m = hpoly_terms - 1; m >= 0; m--)
    hRT = hRT * u + hpoly(n, m);
  hRT += h298(n) * Tinv;
</%pg:macro>

<%pg:macro name="gRTspecies" params="gRT, n, u, Tinv">
  gRT = 0.0;
  for (int m = gpoly_terms - 1; m >= 0; m--)
    gRT = gRT * u + gpoly(n, m);
  gRT += h298(n) * Tinv;
</%pg:macro>

<%pg:macro name="especies" params="e, n, T, u, Tinv">
  fpdtype hRT;
  ${pg.expand("hRTspecies", "hRT, n, u, Tinv")}
  e = (hRT - 1.0) * T * Rsp(n);
</%pg:macro>

<%pg:macro name="hspecies" params="h, mom, Y, state">
  const fpdtype T = state.T, u = log(T), Tinv = 1.0 / T;
  for (int n = 0; n < ${ns}; n++) {
    fpdtype hRT;
    ${pg.expand("hRTspecies", "hRT, n, u, Tinv")}
    h[n] = hRT * T * Ru * MWinv(n);
  }
</%pg:macro>
