<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${chemistry}"/>
<%include file="peregrine.simulator.euler.kernels.solution"/>
## a cell's rates from its solution, and the derivative the rates begin
<%pg:macro name="cellrates" params="rate, Q, state">
  const fpdtype rhoinv = 1.0 / Q[0];
  ${pg.expand("massfrac", "Y, Q, rhoinv", scoped=False)}
  RateState s;
  ${pg.expand("ratestate", "s, Q[0], Y, state.T")}
  ${pg.expand("production", "rate, s")}
</%pg:macro>

## begins dQ: every row written, the fluxes add to it
<%pg:macro name="beginsource" params="dQ, src">
  for (int l = 0; l < 5; l++)
    dQ[l] = 0.0;
  for (int n = 0; n < ${ns} - 1; n++)
    dQ[5 + n] = src[n];
</%pg:macro>
