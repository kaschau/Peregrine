<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.euler.kernels.flux.recon.${recon}"/>
<%include file="${fluxtpls}.function.${primary}"/>
<%include file="peregrine.simulator.euler.kernels.flux.switch.${switch}"/>
<%include file="${fluxtpls}.correction.${correction}"/>
<%pg:kernel name="flux" over="faces"
            S="in [3]"
            F="out [${str(nflux)}]">
  ${pg.expand("sides", "L, R, YL, YR", scoped=False)}
  fpdtype shock, contact;
  ${pg.expand("sensors", "shock, contact")}
  ${pg.expand(primary, "L, R, YL, YR, S, F")}
  ${pg.expand("correction", "L, R, YL, YR, S, shock, contact, F")}
</%pg:kernel>
