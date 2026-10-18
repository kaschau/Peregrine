<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.navstokes.kernels.bcs.wall"/>
<%pg:kernel name="preadvflux" over="halocells">
  const fpdtype wall[3] = {${u}, ${v}, ${w}};
  ${pg.expand("stuckhalo", "wall")}
</%pg:kernel>

<%pg:kernel name="postflux" over="faces" F="out [${str(nflux)}]">
  const fpdtype wall[3] = {${u}, ${v}, ${w}};
  ${pg.expand("adiaflux", "F, wall, 1.0")}
</%pg:kernel>
