<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.chemint.kernels.cell"/>
<%pg:kernel name="source" over="cells"
            Q="in [${str(ne)}]"
            state="in"
            dQ="out [${str(ne)}]">
  fpdtype rate[${ns}];
  ${pg.expand("cellrates", "rate, Q, state")}
  ${pg.expand("beginsource", "dQ, rate")}
</%pg:kernel>
