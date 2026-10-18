<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:alias name="rusanov" file="peregrine.simulator.euler.kernels.flux.function.rusanov" under="plainrusanov"/>
<%include file="${eos}"/>
## reads the plain flux's locals UL, UR and lam, expanded unscoped
<%pg:macro name="rusanov" params="L, R, YL, YR, S, F"
           state="in at L, R">
  ${pg.expand("plainrusanov", "L, R, YL, YR, S, F", scoped=False)}
  fpdtype EpL[2], EpR[2];
  ${pg.expand("eside", "EpL[0], EpL[1], L, YL, stateL, stateR")}
  ${pg.expand("eside", "EpR[0], EpR[1], R, YR, stateL, stateR")}
% for i, row in enumerate((4, ne)):
  F[${row}] = (0.5 * (UR * (EpR[${i}] + R.p) + UL * (EpL[${i}] + L.p) - lam * (EpR[${i}] - EpL[${i}])));
% endfor
</%pg:macro>
