<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${eos}"/>
<%include file="peregrine.simulator.euler.kernels.solution"/>
## the outgoing invariant takes the calorically perfect form on the first cell's gamma and c; the total enthalpy is carried in that convention
<%pg:kernel name="preadvflux" over="halocells"
            S="in [3] at intface"
            outward="in [1] at broadcast"
            Q="in [${str(ne)}] at interior"
            state="in at interior"
            Qhalo="out [${str(ne)}] at halo"
            statehalo="out at halo">
  // the inward unit normal
  const fpdtype Sinv = -outward[0] / sqrt(${pg.dot("S[{i}]", i=3)});
  const fpdtype nx = S[0] * Sinv, ny = S[1] * Sinv, nz = S[2] * Sinv;
  const fpdtype gamma = state.gamma, c = state.c, gm1 = gamma - 1.0;
  const fpdtype rhoinv = 1.0 / Q[0];
  const fpdtype u0[3] = {Q[1] * rhoinv, Q[2] * rhoinv, Q[3] * rhoinv};
  // the entry's composition, every species, the last what the carried ones leave
<% last = "1.0" if not Y else "1.0 - (" + " + ".join(Y) + ")" %>
  const fpdtype Yin[${ns}] = {${", ".join([*Y, last])}};
  const auto Y = [&](const int n) { return Yin[n]; };
  // the entry's total enthalpy: the eos at the total temperature less the eos at the cell's, over the cell's c^2 / (gamma - 1)
  const fpdtype Ttotal = ${Tt};
  MixMom mom;
  ${pg.expand("mixmom", "mom, Y")}
  EosState total, cell;
  ${pg.expand("frompT", "total, mom, Y, state.p, Ttotal")}
  ${pg.expand("frompT", "cell, mom, Y, state.p, state.T")}
  const fpdtype Ht = c * c / gm1 + (total.e + total.p / total.rho) - (cell.e + cell.p / cell.rho);
  const fpdtype Un = u0[0] * nx + u0[1] * ny + u0[2] * nz;
  // the invariant leaving through the inlet, along the outward normal
  const fpdtype Jm = -Un + 2.0 * c / gm1;
  // the larger root of the invariant at the total enthalpy; a negative discriminant, in a transient, takes the double root
  const fpdtype aq = 1.0 + 2.0 / gm1;
  const fpdtype bq = -2.0 * Jm;
  const fpdtype cq = gm1 * (0.5 * Jm * Jm - Ht);
  const fpdtype t1 = -bq / (2.0 * aq);
  const fpdtype t2 = sqrt(fmax(bq * bq - 4.0 * aq * cq, 0.0)) / (2.0 * aq);
  const fpdtype cb = fmax(t1 + t2, t1 - t2);
  const fpdtype Vb = 2.0 * cb / gm1 - Jm;
  const fpdtype Mb = Vb / cb;
  const fpdtype isentropic = 1.0 + 0.5 * gm1 * Mb * Mb;
  const fpdtype p = ${pt} * pow(isentropic, -gamma / gm1), T = Ttotal / isentropic;
  const fpdtype u[3] = {Vb * nx, Vb * ny, Vb * nz};
  ${pg.expand("fromprims", "Qhalo, statehalo, p, T, u, Y")}
</%pg:kernel>
