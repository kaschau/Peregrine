<%namespace name="pg" module="peregrine.codegen.makoutil"/>

<%pg:macro name="kappamix" params="kappa, X, kappasp">
  fpdtype series = 0.0, parallel = 0.0;
  for (int n = 0; n < ${ns}; n++) {
    series += X(n) * kappasp[n];
    parallel += X(n) / kappasp[n];
  }
  kappa = 0.5 * (series + 1.0 / parallel);
</%pg:macro>
