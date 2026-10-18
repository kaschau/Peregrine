<%namespace name="pg" module="peregrine.codegen.makoutil"/>
## within a few percent of Wilke for like molecules; strays for the light species, H2 and H
<%pg:macro name="viscmix" params="mu, X, sqrtmu">
  fpdtype num = 0.0, den = 0.0;
  for (int n = 0; n < ${ns}; n++) {
    const fpdtype w = X(n) * sqrtMW(n);
    num += w * sqrtmu[n] * sqrtmu[n];
    den += w;
  }
  mu = num / den;
</%pg:macro>
