<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<% terms = structs["MixMom"].arrays["cpcoeff"] %>
## every coefficient in the moments is premultiplied by the species' gas constant, so cp and h come out per mass; h's coefficients are cp's integrated, built in the descent as hcoeff[m] = cpcoeff[m] - (m + 1) hcoeff[m + 1]
<%pg:macro name="ideal_cph" params="cp, h, mom, T">
  const fpdtype u = log(T);
  cp = 0.0;
  h = 0.0;
  fpdtype hcoeff = 0.0;
  for (int m = ${terms} - 1; m >= 0; m--) {
    cp = cp * u + mom.cpcoeff[m];
    hcoeff = mom.cpcoeff[m] - (m + 1) * hcoeff;
    h = h * u + hcoeff;
  }
  h = h * T + mom.h298;
</%pg:macro>
