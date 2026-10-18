<%namespace name="pg" module="peregrine.codegen.makoutil"/>
## at least three steps, which settle a warm start at the fits' rounding floor; a cold start goes on to tol, ten times that floor, or the cap; cp and h are left at the T before the last update
<%pg:macro name="newton" params="T, cp, h, mom, e">
  const fpdtype tol = ${1e5 * finfo.eps};
  for (int k = 0; k < 12; k++) {
    ${pg.expand("ideal_cph", "cp, h, mom, T")}
    const fpdtype dT = (e - (h - mom.R * T)) / (cp - mom.R);
    T += dT;
    if (k >= 2 && fabs(dT) <= tol * T) break;
  }
</%pg:macro>
