<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${eos}"/>
<%pg:kernel name="fromplane" over="cells"
            Q="in [${str(ne)}]"
            state="inout">
  const fpdtype rho = Q[0], rhoinv = 1.0 / rho;
  const fpdtype Tguess = state.T;
  ## the carried mass fractions, as the eos reads them
  const auto Y = [&](const int n) { return Q[5 + n] * rhoinv; };
  MixMom mom;
  ${pg.expand("mixmom", "mom, Y")}
  fpdtype p;
  ${pg.expand("pplane", "p, Q, rhoinv, state")}
  EosState thermo;
  ${pg.expand("fromrhop", "thermo, mom, Y, rho, p, Tguess")}
  state.store(thermo);
</%pg:kernel>
