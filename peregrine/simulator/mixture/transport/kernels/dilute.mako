<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${mixrule}"/>
<%include file="peregrine.simulator.mixture.transport.kernels.mixing"/>

<%pg:macro name="dilute" params="mu, kappa, T, lnT, X">
  const fpdtype sqrtT = sqrt(T), sqrtsqrtT = sqrt(sqrtT);
  ## each species' array lives only until its mixing has run
  {
    fpdtype sqrtmu[${ns}];
    for (int n = 0; n < ${ns}; n++) {
      fpdtype fit = 0.0;
      for (int m = mupoly_terms - 1; m >= 0; m--)
        fit = fit * lnT + mupoly(n, m);
      sqrtmu[n] = fit * sqrtsqrtT;
    }
    ${pg.expand("viscmix", "mu, X, sqrtmu")}
  }
  fpdtype kappasp[${ns}];
  for (int n = 0; n < ${ns}; n++) {
    fpdtype fit = 0.0;
    for (int m = kappapoly_terms - 1; m >= 0; m--)
      fit = fit * lnT + kappapoly(n, m);
    kappasp[n] = fit * sqrtT;
  }
  ${pg.expand("kappamix", "kappa, X, kappasp")}
</%pg:macro>
