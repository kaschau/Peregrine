<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.euler.kernels.bcs.wall"/>
<%include file="${eos}"/>
<%include file="peregrine.simulator.euler.kernels.solution"/>
<%pg:kernel name="preadvflux" over="halocells"
            S="in [3] at intface"
            Q="in [${str(ne)}] at mirror"
            state="in at mirror"
            Qhalo="out [${str(ne)}] at halo"
            statehalo="out at halo">
  const fpdtype Sinv = 1.0 / sqrt(${pg.dot("S[{i}]", i=3)});
  const fpdtype nx = S[0] * Sinv, ny = S[1] * Sinv, nz = S[2] * Sinv;
  const fpdtype rhoinv = 1.0 / Q[0];
  const fpdtype uin = Q[1] * rhoinv, vin = Q[2] * rhoinv, win = Q[3] * rhoinv;
  // the velocity's normal part, taken off twice
  const fpdtype un = 2.0 * (uin * nx + vin * ny + win * nz);
  const fpdtype u[3] = {uin - un * nx, vin - un * ny, win - un * nz};
  const fpdtype p = state.p, T = ${T};
  // the image's carried mass fractions, as the eos reads them
  const auto Y = [&](const int n) { return Q[5 + n] * rhoinv; };
  ${pg.expand("fromprims", "Qhalo, statehalo, p, T, u, Y")}
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
