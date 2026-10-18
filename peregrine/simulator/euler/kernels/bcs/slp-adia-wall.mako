<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.euler.kernels.bcs.wall"/>
## the mirror keeps the speed, so the energy and the whole state are the image's as stored; no eos runs
<%pg:kernel name="preadvflux" over="halocells"
            S="in [3] at intface"
            Q="in [${str(ne)}] at mirror"
            state="in at mirror"
            Qhalo="out [${str(ne)}] at halo"
            statehalo="out at halo">
  const fpdtype Sinv = 1.0 / sqrt(${pg.dot("S[{i}]", i=3)});
  const fpdtype nx = S[0] * Sinv, ny = S[1] * Sinv, nz = S[2] * Sinv;
  // the momentum's normal part, taken off twice
  const fpdtype mn = 2.0 * (Q[1] * nx + Q[2] * ny + Q[3] * nz);
  Qhalo[0] = Q[0];
  Qhalo[1] = Q[1] - mn * nx;
  Qhalo[2] = Q[2] - mn * ny;
  Qhalo[3] = Q[3] - mn * nz;
  Qhalo[4] = Q[4];
  for (int n = 0; n < ${ns} - 1; n++)
    Qhalo[5 + n] = Q[5 + n];
  statehalo.copy(state);
</%pg:kernel>

## the wall writes the flux through its faces itself, from the first cell inward, over the flux function's
<%pg:kernel name="postflux" over="faces"
            S="in [3]"
            outward="in [1] at broadcast"
            Q="in [${str(ne)}] at interior"
            state="in at interior"
            F="out [${str(nflux)}]">
  fpdtype pw;
  ${pg.expand("wallpressure", "pw, Q, state, outward[0], S")}
  ${pg.expand("wallflux", "F, pw, S")}
</%pg:kernel>
