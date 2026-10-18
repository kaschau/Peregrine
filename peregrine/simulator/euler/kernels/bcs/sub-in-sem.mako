<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${eos}"/>
<%include file="peregrine.simulator.euler.kernels.solution"/>
## semcoeffs are the signal's cubic on the interface from tbreak, already in grid components
<%pg:kernel name="preadvflux" over="halocells"
            semcoeffs="in [3, 4] at plane"
            tbreak="in [1] at broadcast"
            tme="in"
            state="in at interior, mirror1"
            Qhalo="out [${str(ne)}] at halo"
            statehalo="out at halo">
  const fpdtype dt = tme - tbreak[0];
  fpdtype u[3];
  for (int v = 0; v < 3; v++)
    u[v] = ((semcoeffs[v][0] * dt + semcoeffs[v][1]) * dt + semcoeffs[v][2]) * dt + semcoeffs[v][3];
  const fpdtype p = 2.0 * stateinterior.p - statemirror1.p, T = ${T};
  ## the entry's composition, every species, the last what the carried ones leave
<% last = "1.0" if not Y else "1.0 - (" + " + ".join(Y) + ")" %>
  const fpdtype Yin[${ns}] = {${", ".join([*Y, last])}};
  const auto Y = [&](const int n) { return Yin[n]; };
  ${pg.expand("fromprims", "Qhalo, statehalo, p, T, u, Y")}
</%pg:kernel>
