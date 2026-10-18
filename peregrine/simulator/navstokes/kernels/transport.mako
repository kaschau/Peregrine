<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${eos}"/>
<%include file="${trans}"/>
<%include file="${diffusion}"/>
<%pg:kernel name="transport" over="cells"
            Q="in [${str(ne)}]"
            state="in"
            trstate="out [${str(2 + ns)}]"
            hsp="out [${str(ns)}]">
  const fpdtype rho = Q[0], rhoinv = 1.0 / rho;
  fpdtype Ylast = 1.0;
  for (int n = 0; n < ${ns} - 1; n++)
    Ylast -= Q[5 + n] * rhoinv;
  Ylast = fmax(Ylast, 0.0);
  const auto Y = [&](const int n) { return n < ${ns} - 1 ? Q[5 + n] * rhoinv : Ylast; };
  fpdtype MWmix;
  ${pg.expand("molmass", "MWmix, Y")}
  const auto X = [&](const int n) { return Y(n) * MWinv(n) * MWmix; };
  const fpdtype p = state.p, T = state.T, lnT = log(T);
  // the mixture's moments once, for the transport model and the enthalpies
  MixMom mom;
  ${pg.expand("mixmom", "mom, Y")}
  fpdtype mu, kappa;
  ${pg.expand("transport", "mu, kappa, rho, T, lnT, X, Y, mom")}
  trstate[0] = mu;
  trstate[1] = kappa;
  fpdtype D[${ns}];
  ${pg.expand("diffcoeffs", "D, p, T, lnT, rhoinv, state.cp, kappa, MWmix, X")}
  for (int n = 0; n < ${ns}; n++)
    trstate[2 + n] = D[n];
  fpdtype h[${ns}];
  ${pg.expand("hspecies", "h, mom, Y, state")}
  for (int n = 0; n < ${ns}; n++)
    hsp[n] = h[n];
</%pg:kernel>
