<%namespace name="pg" module="peregrine.codegen.makoutil"/>
## stepped as the real gas's newton, at least three then to tol or the cap; the residual is nearly linear in T and converges from anywhere
<%pg:macro name="newtonp" params="T, c, Y, p, invvmb, invD">
  const fpdtype tol = ${1e5 * finfo.eps};
  for (int k = 0; k < 12; k++) {
    ${pg.expand("attraction", "c, Y, T")}
    const fpdtype f = Ru * T * invvmb - c.am * invD - p;
    const fpdtype dpdT = Ru * invvmb - c.dam * invD;
    const fpdtype dT = f / dpdT;
    T -= dT;
    if (k >= 2 && fabs(dT) <= tol * T) break;
  }
</%pg:macro>
