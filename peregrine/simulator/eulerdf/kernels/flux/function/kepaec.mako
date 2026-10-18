<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:alias name="kepaec" file="peregrine.simulator.euler.kernels.flux.function.kepaec" under="plainkepaec"/>
<%include file="${eos}"/>
## reads the plain flux's locals C, Kj and Pj, expanded unscoped
<%pg:macro name="kepaec" params="L, R, YL, YR, S, F"
           state="in at L, R">
  ${pg.expand("plainkepaec", "L, R, YL, YR, S, F", scoped=False)}
  const fpdtype rhoinvL = 1.0 / L.rho, rhoinvR = 1.0 / R.rho;
  fpdtype cL[2], cR[2];
  ${pg.expand("ecomp", "cL[0], cL[1], YL, stateL, stateR")}
  ${pg.expand("ecomp", "cR[0], cR[1], YR, stateL, stateR")}
% for i, (row, plane) in enumerate(((4, "stateL"), (ne, "stateR"))):
  {
    fpdtype tL, tR;
    ${pg.expand("ethermal", f"tL, L.p, rhoinvL, {plane}")}
    ${pg.expand("ethermal", f"tR, R.p, rhoinvR, {plane}")}
    ## the thermal part's harmonic mean is theta p + kappa over the mean density at one pressure, so a contact at rest moves no energy the plane cannot place
    F[${row}] = ((${plane}.eps + 0.5 * (cL[${i}] + cR[${i}]) + 2.0 * tL * tR / (tL + tR)) * C + Kj + Pj);
  }
% endfor
</%pg:macro>
