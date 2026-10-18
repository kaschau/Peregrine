<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:alias name="preadvflux" file="peregrine.simulator.euler.kernels.bcs.slp-isot-wall"/>
<%include file="peregrine.simulator.navstokes.kernels.bcs.wall"/>
<%pg:kernel name="postflux" over="faces" F="out [${str(nflux)}]">
  const fpdtype still[3] = {0.0, 0.0, 0.0}, Tw = ${T};
  ${pg.expand("isotflux", "F, still, 0.0, Tw")}
</%pg:kernel>
