<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.chemint.kernels.cell"/>
<%pg:kernel name="production" over="cells"
            Q="in [${str(ne)}]"
            state="in"
            omega="out [${str(ns)}]">
  fpdtype rate[${ns}];
  ${pg.expand("cellrates", "rate, Q, state")}
  for (int n = 0; n < ${ns}; n++)
    omega[n] = rate[n];
</%pg:kernel>
