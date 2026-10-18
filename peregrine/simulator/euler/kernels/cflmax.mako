<%namespace name="pg" module="peregrine.codegen.makoutil"/>
## the acoustic length is 1 / sqrt(sum 1 / d^2), the oblique mode a central scheme sees, which the largest direction alone would miss
<%pg:kernel name="cflmax" over="cells" reduce="cfl max [3]"
            Q="in [${str(ne)}]"
            state="in"
            dijk="in [3]"
            iS="in [3] at low, high"
            jS="in [3] at low, high"
            kS="in [3] at low, high">
  // the cell lengths; an axis not marched in is infinitely long
  const fpdtype di = dijk[0], dj = dijk[1], dk = dijk[2];
  const fpdtype rhoinv = 1.0 / Q[0];
  const fpdtype u = Q[1] * rhoinv, v = Q[2] * rhoinv, w = Q[3] * rhoinv;
  // each direction's normal velocity, the normal the mean of the cell's two faces' unit normals
% for d, L, H in (("I", "iSlow", "iShigh"), ("J", "jSlow", "jShigh"), ("K", "kSlow", "kShigh")):
  fpdtype u${d};
  {
    const fpdtype S0inv = 1.0 / sqrt(${pg.dot(L + "[{i}]", i=3)});
    const fpdtype S1inv = 1.0 / sqrt(${pg.dot(H + "[{i}]", i=3)});
    const fpdtype nx = 0.5 * (${L}[0] * S0inv + ${H}[0] * S1inv);
    const fpdtype ny = 0.5 * (${L}[1] * S0inv + ${H}[1] * S1inv);
    const fpdtype nz = 0.5 * (${L}[2] * S0inv + ${H}[2] * S1inv);
    u${d} = fabs(nx * u + ny * v + nz * w);
  }
% endfor
  const fpdtype c = state.c;
  // an infinite length contributes nothing
  const fpdtype acoustic = c * sqrt(1.0 / (di * di) + 1.0 / (dj * dj) + 1.0 / (dk * dk));
  const fpdtype convective = uI / di + uJ / dj + uK / dk;
  cfl[0] = fmax(cfl[0], acoustic);
  cfl[1] = fmax(cfl[1], convective);
  cfl[2] = fmax(cfl[2], acoustic + convective);
</%pg:kernel>
