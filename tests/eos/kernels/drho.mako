<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${eos}"/>
<%pg:kernel name="drho" over="cells"
            Q="in [${str(ne)}]"
            state="in"
            qj="out [${str(ne - 3)}]">
  const fpdtype rho = Q[0], rhoinv = 1.0 / rho;
  const auto Y = [&](const int n) { return Q[5 + n] * rhoinv; };
  MixMom mom;
  ${pg.expand("mixmom", "mom, Y")}
  const fpdtype p = state.p, T = state.T;
  fpdtype rho_p, rho_T, rho_Y[${max(ns - 1, 1)}];
  ${pg.expand("drho", "rho_p, rho_T, rho_Y, mom, Y, p, T, rho")}
  qj[0] = rho_p;
  qj[1] = rho_T;
  for (int n = 0; n < ${ns} - 1; n++)
    qj[2 + n] = rho_Y[n];
</%pg:kernel>
