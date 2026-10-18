<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:alias name="preadvflux" file="peregrine.simulator.navstokes.kernels.bcs.mov-adia-wall"/>
<%include file="peregrine.simulator.navstokes.kernels.bcs.wall"/>
## nothing is advected through a wall, so the energy row is the same in either plane
<%pg:kernel name="postflux" over="faces" F="out [${str(nflux)}]">
  const fpdtype wall[3] = {${u}, ${v}, ${w}};
  ${pg.expand("adiaflux", "F, wall, 1.0")}
  F[${ne}] = F[4];
</%pg:kernel>
