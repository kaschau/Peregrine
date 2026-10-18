<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${mixrule}"/>
<%include file="peregrine.simulator.mixture.transport.kernels.mixing"/>

<%pg:macro name="transport" params="mu, kappa, rho, T, lnT, X, Y, mom">
  {
    fpdtype sqrtmu[${ns}];
    for (int n = 0; n < ${ns}; n++)
      sqrtmu[n] = sqrt(mu0(n));
    ${pg.expand("viscmix", "mu, X, sqrtmu")}
  }
  fpdtype kappasp[${ns}];
  for (int n = 0; n < ${ns}; n++)
    kappasp[n] = kappa0(n);
  ${pg.expand("kappamix", "kappa, X, kappasp")}
</%pg:macro>
