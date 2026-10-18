<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:alias name="ausmup" file="peregrine.simulator.euler.kernels.flux.function.ausmup" under="plainausmup"/>
<%include file="${eos}"/>
## reads the plain flux's locals mdot, U and left, expanded unscoped
<%pg:macro name="ausmup" params="L, R, YL, YR, S, F"
           state="in at L, R">
  ${pg.expand("plainausmup", "L, R, YL, YR, S, F", scoped=False)}
  const fpdtype rhoinvU = 1.0 / U.rho;
  const auto YU = [&](const int n) { return left ? YL(n) : YR(n); };
  fpdtype Ep[2];
  ${pg.expand("eside", "Ep[0], Ep[1], U, YU, stateL, stateR")}
% for i, row in enumerate((4, ne)):
  F[${row}] = (mdot * (Ep[${i}] + U.p) * rhoinvU);
% endfor
</%pg:macro>
