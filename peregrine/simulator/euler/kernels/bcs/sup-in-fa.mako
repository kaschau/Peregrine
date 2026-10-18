<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${eos}"/>
<%include file="peregrine.simulator.euler.kernels.solution"/>
<%pg:kernel name="preadvflux" over="halocells"
            Qhalo="out [${str(ne)}] at halo"
            statehalo="out at halo">
  const fpdtype p = ${p}, T = ${T};
  const fpdtype u[3] = {${u}, ${v}, ${w}};
  ## the entry's composition, every species, the last what the carried ones leave
<% last = "1.0" if not Y else "1.0 - (" + " + ".join(Y) + ")" %>
  const fpdtype Yin[${ns}] = {${", ".join([*Y, last])}};
  const auto Y = [&](const int n) { return Yin[n]; };
  ${pg.expand("fromprims", "Qhalo, statehalo, p, T, u, Y")}
</%pg:kernel>
