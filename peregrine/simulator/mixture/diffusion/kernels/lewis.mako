<%namespace name="pg" module="peregrine.codegen.makoutil"/>

<%pg:macro name="diffcoeffs" params="D, p, T, lnT, rhoinv, cp, kappa, MWmix, X">
  const fpdtype kappaoverrhocp = kappa * rhoinv / cp;
  for (int n = 0; n < ${ns}; n++)
    D[n] = kappaoverrhocp / lewis(n);
</%pg:macro>
