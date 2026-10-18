<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:alias name="preadvflux" file="peregrine.simulator.euler.kernels.bcs.slp-adia-wall"/>
<%include file="peregrine.simulator.navstokes.kernels.bcs.wall"/>
<%pg:kernel name="postflux" over="faces" F="out [${str(nflux)}]">
  const fpdtype still[3] = {0.0, 0.0, 0.0};
  ${pg.expand("adiaflux", "F, still, 0.0")}
</%pg:kernel>
