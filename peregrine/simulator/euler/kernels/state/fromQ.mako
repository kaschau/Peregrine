<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${eos}"/>
<%include file="peregrine.simulator.euler.kernels.solution"/>
## state is inout only for the cell's last T, an iterative eos's starting guess
<%pg:kernel name="fromQ" over="cells"
            Q="in [${str(ne)}]"
            state="inout">
  const fpdtype rho = Q[0], rhoinv = 1.0 / rho;
  const fpdtype Tguess = state.T;
  fpdtype e;
  ${pg.expand("efromQ", "e, Q, rhoinv")}
  ## the carried mass fractions, as the eos reads them
  const auto Y = [&](const int n) { return Q[5 + n] * rhoinv; };
  MixMom mom;
  ${pg.expand("mixmom", "mom, Y")}
  EosState thermo;
  ${pg.expand("fromrhoE", "thermo, mom, Y, rho, e, Tguess")}
  state.store(thermo);
</%pg:kernel>
