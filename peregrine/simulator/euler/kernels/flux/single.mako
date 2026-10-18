<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.euler.kernels.flux.recon.${recon}"/>
<%include file="${fluxtpls}.function.${primary}"/>
<%pg:kernel name="flux" over="faces"
            S="in [3]"
            F="out [${str(nflux)}]">
  ${pg.expand("sides", "L, R, YL, YR", scoped=False)}
  ${pg.expand(primary, "L, R, YL, YR, S, F")}
</%pg:kernel>
