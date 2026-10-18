<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${eos}"/>
<%include file="peregrine.simulator.euler.kernels.solution"/>
<%pg:kernel name="fromprims" over="cells"
            prims="in [${str(ne)}]"
            Q="out [${str(ne)}]"
            state="out">
  const fpdtype p = prims[0], T = prims[4];
  const fpdtype u[3] = {prims[1], prims[2], prims[3]};
  ## the carried mass fractions, as the eos reads them
  const auto Y = [&](const int n) { return prims[5 + n]; };
  ${pg.expand("fromprims", "Q, state, p, T, u, Y")}
</%pg:kernel>
