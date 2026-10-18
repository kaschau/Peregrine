<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${eos}"/>
<%pg:kernel name="hpartial" over="cells"
            Q="in [${str(ne)}]"
            state="in"
            hi="out [${str(ns)}]">
  const fpdtype rho = Q[0], rhoinv = 1.0 / rho;
  const auto Y = [&](const int n) { return Q[5 + n] * rhoinv; };
  MixMom mom;
  ${pg.expand("mixmom", "mom, Y")}
  fpdtype h[${ns}];
  ${pg.expand("hspecies", "h, mom, Y, state")}
  for (int n = 0; n < ${ns}; n++)
    hi[n] = h[n];
</%pg:kernel>
