<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:alias name="preadvflux" file="peregrine.simulator.navstokes.kernels.bcs.slp-isot-wall"/>
<%include file="peregrine.simulator.navstokes.kernels.bcs.wall"/>
## nothing is advected through a wall, so the energy row is the same in either plane
<%pg:kernel name="postflux" over="faces" F="out [${str(nflux)}]">
  const fpdtype still[3] = {0.0, 0.0, 0.0}, Tw = ${T};
  ${pg.expand("isotflux", "F, still, 0.0, Tw")}
  F[${ne}] = F[4];
</%pg:kernel>
