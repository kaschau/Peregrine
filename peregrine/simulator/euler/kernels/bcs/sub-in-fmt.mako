<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${eos}"/>
<%include file="peregrine.simulator.euler.kernels.solution"/>
## the halo velocity makes the face's averages of density and of velocity carry the entry's mass flux
<%pg:kernel name="preadvflux" over="halocells"
            S="in [3] at intface"
            outward="in [1] at broadcast"
            Q="in [${str(ne)}] at interior"
            state="in at interior, mirror1"
            Qhalo="out [${str(ne)}] at halo"
            statehalo="out at halo">
  // the inward unit normal
  const fpdtype Sinv = -outward[0] / sqrt(${pg.dot("S[{i}]", i=3)});
  const fpdtype nx = S[0] * Sinv, ny = S[1] * Sinv, nz = S[2] * Sinv;
  const fpdtype p = 2.0 * stateinterior.p - statemirror1.p, T = ${T};
  // the entry's composition, every species, the last what the carried ones leave
<% last = "1.0" if not Y else "1.0 - (" + " + ".join(Y) + ")" %>
  const fpdtype Yin[${ns}] = {${", ".join([*Y, last])}};
  const auto Y = [&](const int n) { return Yin[n]; };
  MixMom mom;
  ${pg.expand("mixmom", "mom, Y")}
  EosState thermo;
  ${pg.expand("frompT", "thermo, mom, Y, p, T")}
  const fpdtype rhoinv = 1.0 / Q[0];
  const fpdtype scale = 4.0 * ${mflux} / (Q[0] + thermo.rho);
  const fpdtype u[3] = {nx * scale - Q[1] * rhoinv, ny * scale - Q[2] * rhoinv, nz * scale - Q[3] * rhoinv};
  ${pg.expand("fromthermo", "Qhalo, thermo, u")}
  for (int n = 0; n < ${ns} - 1; n++)
    Qhalo[5 + n] = thermo.rho * Y(n);
  statehalo.store(thermo);
</%pg:kernel>
