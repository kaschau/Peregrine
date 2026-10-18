<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${eos}"/>
<%include file="peregrine.simulator.euler.kernels.solution"/>
## where the first cell's flow comes back in, the velocity is mirrored about the face like a slip wall's
<%pg:kernel name="preadvflux" over="halocells"
            S="in [3] at intface"
            outward="in [1] at broadcast"
            Q="in [${str(ne)}] at interior, mirror, mirror1"
            state="in at interior"
            Qhalo="out [${str(ne)}] at halo"
            statehalo="out at halo">
  // the outward unit normal
  const fpdtype Sinv = outward[0] / sqrt(${pg.dot("S[{i}]", i=3)});
  const fpdtype nx = S[0] * Sinv, ny = S[1] * Sinv, nz = S[2] * Sinv;
  const fpdtype rhoinv = 1.0 / Qinterior[0];
  const fpdtype p = ${p}, T = state.T;
  const fpdtype u0[3] = {Qinterior[1] * rhoinv, Qinterior[2] * rhoinv, Qinterior[3] * rhoinv};
  const fpdtype un = u0[0] * nx + u0[1] * ny + u0[2] * nz;
  fpdtype u[3];
  if (un > 0.0) {
    const fpdtype rhoinv1 = 1.0 / Qmirror1[0];
    for (int d = 0; d < 3; d++)
      u[d] = 2.0 * u0[d] - Qmirror1[1 + d] * rhoinv1;
  } else {
    const fpdtype rhoinvm = 1.0 / Qmirror[0];
    const fpdtype um[3] = {Qmirror[1] * rhoinvm, Qmirror[2] * rhoinvm, Qmirror[3] * rhoinvm};
    const fpdtype umn = 2.0 * (um[0] * nx + um[1] * ny + um[2] * nz);
    u[0] = um[0] - umn * nx;
    u[1] = um[1] - umn * ny;
    u[2] = um[2] - umn * nz;
  }
  // the first cell's carried mass fractions, the last what they leave
  const auto Y = [&](const int n) -> fpdtype {
    if (n < ${ns} - 1)
      return Qinterior[5 + n] * rhoinv;
    fpdtype last = 1.0;
    for (int m = 0; m < ${ns} - 1; m++)
      last -= Qinterior[5 + m] * rhoinv;
    return last;
  };
  ${pg.expand("fromprims", "Qhalo, statehalo, p, T, u, Y")}
</%pg:kernel>
