<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${eos}"/>
<%include file="peregrine.simulator.euler.kernels.solution"/>
## the pressure is capped at the interior's so a wave can only leave; the floors on p and T, a hundredth of the interior's, are a safety net for a transient so the halo keeps a density
<%pg:kernel name="preadvflux" over="halocells"
            S="in [3] at intface"
            outward="in [1] at broadcast"
            Q="in [${str(ne)}] at interior, mirror, mirror1"
            state="in at interior, mirror1"
            Qhalo="out [${str(ne)}] at halo"
            statehalo="out at halo">
  // the outward unit normal
  const fpdtype Sinv = outward[0] / sqrt(${pg.dot("S[{i}]", i=3)});
  const fpdtype nx = S[0] * Sinv, ny = S[1] * Sinv, nz = S[2] * Sinv;
  const fpdtype rhoinv = 1.0 / Qinterior[0], rhoinv1 = 1.0 / Qmirror1[0];
  const fpdtype least = 0.01;
  const fpdtype p0 = stateinterior.p, T0 = stateinterior.T;
  const fpdtype p = fmin(fmax(least * p0, 2.0 * p0 - statemirror1.p), p0);
  const fpdtype T = fmax(least * T0, 2.0 * T0 - statemirror1.T);
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
  // the carried mass fractions extrapolated and kept in [0, 1], the last what they leave
  const auto Yn = [&](const int n) -> fpdtype {
    const fpdtype y = 2.0 * Qinterior[5 + n] * rhoinv - Qmirror1[5 + n] * rhoinv1;
    return fmax(0.0, fmin(1.0, y));
  };
  const auto Y = [&](const int n) -> fpdtype {
    if (n < ${ns} - 1)
      return Yn(n);
    fpdtype last = 1.0;
    for (int m = 0; m < ${ns} - 1; m++)
      last -= Yn(m);
    return last;
  };
  ${pg.expand("fromprims", "Qhalo, statehalo, p, T, u, Y")}
</%pg:kernel>
