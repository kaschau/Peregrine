<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.navstokes.kernels.bcs.wall"/>
<%pg:kernel name="preadvflux" over="halocells">
  const fpdtype still[3] = {0.0, 0.0, 0.0};
  ${pg.expand("stuckhalo", "still")}
</%pg:kernel>

<%pg:kernel name="postflux" over="faces" F="out [${str(nflux)}]">
  const fpdtype still[3] = {0.0, 0.0, 0.0};
  ${pg.expand("adiaflux", "F, still, 1.0")}
</%pg:kernel>
