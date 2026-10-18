<%namespace name="pg" module="peregrine.codegen.makoutil"/>
## at least three steps, which settle a warm start at the enthalpy's rounding floor; a cold start (a restart's padded halo, ignition within a step) goes on to tol, ten times that floor, or the cap
<%pg:macro name="newton" params="T, cp_ideal, h_ideal, c, Y, mom, e, invMWmix">
  const fpdtype tol = ${1e5 * finfo.eps};
  for (int k = 0; k < 12; k++) {
    ${pg.expand("attraction", "c, Y, T")}
    ${pg.expand("ideal_cph", "cp_ideal, h_ideal, mom, T")}
    const fpdtype app = 0.5 * (c.S2 * c.S2 - c.dam) / T;
    const fpdtype f = h_ideal - mom.R * T - (c.am - T * c.dam) * c.I * invMWmix - e;
    const fpdtype cv = cp_ideal - mom.R + T * app * c.I * invMWmix;
    const fpdtype dT = f / cv;
    T -= dT;
    if (k >= 2 && fabs(dT) <= tol * T) break;
  }
</%pg:macro>
