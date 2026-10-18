<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.euler.kernels.bcs.wall"/>
<%include file="peregrine.simulator.euler.kernels.solution"/>
<%include file="${eos}"/>
<%include file="${trans}"/>

<%pg:macro name="stuckhalo" params="wall"
           Q="in [${str(ne)}] at mirror"
           state="in at mirror"
           Qhalo="out [${str(ne)}] at halo"
           statehalo="out at halo">
  const fpdtype rho = Q[0], rhoinv = 1.0 / rho;
  fpdtype ke = 0.0, kehalo = 0.0;
  for (int d = 0; d < 3; d++) {
    const fpdtype u = Q[1 + d] * rhoinv, uh = 2.0 * wall[d] - u;
    Qhalo[1 + d] = rho * uh;
    ke += u * u;
    kehalo += uh * uh;
  }
  Qhalo[0] = rho;
  Qhalo[4] = Q[4] + 0.5 * rho * (kehalo - ke);
  for (int n = 0; n < ${ns} - 1; n++)
    Qhalo[5 + n] = Q[5 + n];
  statehalo.copy(state);
</%pg:macro>

<%pg:macro name="stuckhaloisot" params="wall, Tw"
           Q="in [${str(ne)}] at mirror"
           state="in at mirror"
           Qhalo="out [${str(ne)}] at halo"
           statehalo="out at halo">
  const fpdtype rhoinv = 1.0 / Q[0];
  const fpdtype u[3] = {2.0 * wall[0] - Q[1] * rhoinv, 2.0 * wall[1] - Q[2] * rhoinv, 2.0 * wall[2] - Q[3] * rhoinv};
  const fpdtype p = state.p, T = Tw;
  const auto Y = [&](const int n) { return Q[5 + n] * rhoinv; };
  ${pg.expand("fromprims", "Qhalo, statehalo, p, T, u, Y")}
</%pg:macro>

<%pg:macro name="wallcoeffs" params="mu, kappa, Tw"
           Q="in [${str(ne)}] at interior"
           state="in at interior">
  const fpdtype rhoinv = 1.0 / Q[0];
  fpdtype Ylast = 1.0;
  for (int n = 0; n < ${ns} - 1; n++)
    Ylast -= Q[5 + n] * rhoinv;
  Ylast = fmax(Ylast, 0.0);
  const auto Y = [&](const int n) { return n < ${ns} - 1 ? Q[5 + n] * rhoinv : Ylast; };
  MixMom mom;
  ${pg.expand("mixmom", "mom, Y")}
  EosState thermo;
  ${pg.expand("frompT", "thermo, mom, Y, state.p, Tw")}
  fpdtype MWmix;
  ${pg.expand("molmass", "MWmix, Y")}
  const auto X = [&](const int n) { return Y(n) * MWinv(n) * MWmix; };
  const fpdtype T = Tw, lnT = log(T);
  ${pg.expand("transport", "mu, kappa, thermo.rho, T, lnT, X, Y, mom")}
</%pg:macro>

## only the velocity's normal derivative enters the stress, a two-point gradient from the cell to the face; (dun . n) n / 3 is the transpose and Stokes' bulk terms combined
<%pg:macro name="wallfaceflux" params="F, wall, stuck, mu, kappa, dT"
           S="in [3]"
           outward="in [1] at broadcast"
           xf="in [3]"
           Q="in [${str(ne)}] at interior"
           state="in at interior"
           xcc="in [3] at interior">
  fpdtype pw;
  ${pg.expand("wallpressure", "pw, Q, state, outward[0], S")}
  ${pg.expand("wallflux", "F, pw, S")}
  const fpdtype Sabs = sqrt(${pg.dot("S[{i}]", i=3)}), Sinv = 1.0 / Sabs;
  const fpdtype n[3] = {S[0] * Sinv, S[1] * Sinv, S[2] * Sinv};
  const fpdtype dn = fabs(${pg.dot("(xcc[{i}] - xf[{i}])", "n[{i}]", i=3)});
  const fpdtype rhoinv = 1.0 / Q[0];
  fpdtype dun[3];
  for (int d = 0; d < 3; d++)
    dun[d] = outward[0] * (wall[d] - Q[1 + d] * rhoinv) / dn;
  const fpdtype div = ${pg.dot("dun[{i}]", "n[{i}]", i=3)};
  // every component where the flow is held, the normal one alone where it slips
  for (int d = 0; d < 3; d++)
    dun[d] = div * n[d] + stuck * (dun[d] - div * n[d]);
  fpdtype work = 0.0;
  for (int d = 0; d < 3; d++) {
    const fpdtype tS = -mu * Sabs * (dun[d] + div * n[d] / 3.0);
    F[1 + d] += tS;
    work += wall[d] * tS;
  }
  F[4] += work - kappa * outward[0] * (dT) / dn * Sabs;
</%pg:macro>

<%pg:macro name="adiaflux" params="F, wall, stuck"
           trstate="in [${str(2 + ns)}] at interior">
  ${pg.expand("wallfaceflux", "F, wall, stuck, trstate[0], trstate[1], 0.0")}
</%pg:macro>

<%pg:macro name="isotflux" params="F, wall, stuck, Tw">
  fpdtype mu, kappa;
  ${pg.expand("wallcoeffs", "mu, kappa, Tw")}
  ${pg.expand("wallfaceflux", "F, wall, stuck, mu, kappa, Tw - state.T")}
</%pg:macro>
