<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.euler.kernels.flux.recon.${limiter}"/>
<%include file="${eos}"/>
## phi is the limited slope across the face, formed without the ratio's division: the limiter takes the two differences carried to dp's sign and the sign goes back on; where the slope across the face is none, phi is none
<%pg:macro name="phi" params="phi, dm, dp">
  phi = 0.0;
  if ((dp) != 0.0) {
    const fpdtype s = copysign(1.0, dp), a = s * (dm), b = fabs(dp);
    fpdtype bphi;
    ${pg.expand("limiter", "bphi, a, b")}
    phi = s * bphi;
  }
</%pg:macro>

<%pg:macro name="sides" params="L, R, YL, YR"
           Q="in [${str(ne)}] at LL, L, R, RR"
           state="in at LL, L, R, RR">
  FaceState L, R;
  const fpdtype rhoinvLL = stateLL.rhoinv, rhoinvL = stateL.rhoinv, rhoinvR = stateR.rhoinv, rhoinvRR = stateRR.rhoinv;
  // both sides take the upwind difference, away from the face, against the one across it, so an upwind-biased limiter biases each side upwind
  const auto left = [&](const fpdtype vLL, const fpdtype vL, const fpdtype vR) {
    fpdtype phi;
    ${pg.expand("phi", "phi, vL - vLL, vR - vL")}
    return vL + 0.5 * phi;
  };
  const auto right = [&](const fpdtype vL, const fpdtype vR, const fpdtype vRR) {
    fpdtype phi;
    ${pg.expand("phi", "phi, vRR - vR, vR - vL")}
    return vR - 0.5 * phi;
  };
  const fpdtype pL = left(stateLL.p, stateL.p, stateR.p), pR = right(stateL.p, stateR.p, stateRR.p);
  const fpdtype TL = left(stateLL.T, stateL.T, stateR.T), TR = right(stateL.T, stateR.T, stateRR.T);
  fpdtype YLs[${max(ns - 1, 1)}], YRs[${max(ns - 1, 1)}];
  for (int n = 0; n < ${ns} - 1; n++) {
    YLs[n] = left(QLL[5 + n] * rhoinvLL, QL[5 + n] * rhoinvL, QR[5 + n] * rhoinvR);
    YRs[n] = right(QL[5 + n] * rhoinvL, QR[5 + n] * rhoinvR, QRR[5 + n] * rhoinvRR);
  }
  const auto YL = [&](const int n) { return YLs[n]; };
  const auto YR = [&](const int n) { return YRs[n]; };
  // the eos fills each side from its face p, T and Y, so a side is a consistent state whatever the limiters did
% for side, cells in (("L", ("LL", "L", "R")), ("R", ("L", "R", "RR"))):
  {
    MixMom mom;
    ${pg.expand("mixmom", f"mom, Y{side}")}
    ${pg.expand("frompT", f"{side}, mom, Y{side}, p{side}, T{side}")}
% for d, u in enumerate("uvw"):
    ${side}.${u} = ${"left" if side == "L" else "right"}(${", ".join(f"Q{c}[{1 + d}] * rhoinv{c}" for c in cells)});
% endfor
  }
% endfor
</%pg:macro>
