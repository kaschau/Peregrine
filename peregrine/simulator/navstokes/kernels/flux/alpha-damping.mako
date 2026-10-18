<%namespace name="pg" module="peregrine.codegen.makoutil"/>
## the face gradients by alpha damping: Nishikawa, Robust numerical fluxes for unrealizable states, JCP 408 (2020)
<%pg:macro name="alphadamping" params="S, F"
           xf="in [3]"
           xcc="in [3] at L, R"
           Q="in [${str(ne)}] at L, R"
           state="in at L, R"
           trstate="in [${str(2 + ns)}] at L, R"
           hsp="in [${str(ns)}] at L, R"
           grads="in [${str(ngrad)}, 3] at L, R">
  const fpdtype alpha = ${values.get("alpha", 1.0)}, bulkvisc = 0.0;
  // the face's area and unit normal
  const fpdtype Sabs = sqrt(${pg.dot("S[{i}]", i=3)}), Sinv = 1.0 / Sabs;
  const fpdtype nx = S[0] * Sinv, ny = S[1] * Sinv, nz = S[2] * Sinv;
  const fpdtype mu = 0.5 * (trstateR[0] + trstateL[0]);
  const fpdtype kappa = 0.5 * (trstateR[1] + trstateL[1]);
  const fpdtype lambda = bulkvisc - 2.0 / 3.0 * mu;
  const fpdtype rhoinvL = stateL.rhoinv, rhoinvR = stateR.rhoinv;
  const fpdtype uL = QL[1] * rhoinvL, vL = QL[2] * rhoinvL, wL = QL[3] * rhoinvL;
  const fpdtype uR = QR[1] * rhoinvR, vR = QR[2] * rhoinvR, wR = QR[3] * rhoinvR;
  // the damping over the cells' separation along the normal, and each cell's reach to the face
  const fpdtype ex = xccR[0] - xccL[0], ey = xccR[1] - xccL[1], ez = xccR[2] - xccL[2];
  const fpdtype en0 = ex * nx, en1 = ey * ny, en2 = ez * nz;
  const fpdtype dampinv = alpha / sqrt(en0 * en0 + en1 * en1 + en2 * en2);
  const fpdtype dampx = dampinv * nx, dampy = dampinv * ny, dampz = dampinv * nz;
  const fpdtype xLx = xf[0] - xccL[0], xLy = xf[1] - xccL[1], xLz = xf[2] - xccL[2];
  const fpdtype xRx = xf[0] - xccR[0], xRy = xf[1] - xccR[1], xRz = xf[2] - xccR[2];
  // the face gradient of row l: the sides' average, damped toward the jump between the values each side extrapolates to the face
  const auto facegrad = [&](const int l, const fpdtype phiL, const fpdtype phiR, fpdtype &gx, fpdtype &gy, fpdtype &gz) {
    const fpdtype gLx = gradsL[l][0], gLy = gradsL[l][1], gLz = gradsL[l][2];
    const fpdtype gRx = gradsR[l][0], gRy = gradsR[l][1], gRz = gradsR[l][2];
    const fpdtype wl = phiL + gLx * xLx + gLy * xLy + gLz * xLz;
    const fpdtype wr = phiR + gRx * xRx + gRy * xRy + gRz * xRz;
    const fpdtype jump = wr - wl;
    gx = 0.5 * (gRx + gLx) + dampx * jump;
    gy = 0.5 * (gRy + gLy) + dampy * jump;
    gz = 0.5 * (gRz + gLz) + dampz * jump;
  };
  fpdtype dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz;
  facegrad(0, uL, uR, dudx, dudy, dudz);
  facegrad(1, vL, vR, dvdx, dvdy, dvdz);
  facegrad(2, wL, wR, dwdx, dwdy, dwdz);
  const fpdtype div = dudx + dvdy + dwdz;
  const fpdtype txx = -2.0 * mu * dudx - lambda * div;
  const fpdtype txy = -mu * (dvdx + dudy);
  const fpdtype txz = -mu * (dwdx + dudz);
  const fpdtype tyy = -2.0 * mu * dvdy - lambda * div;
  const fpdtype tyz = -mu * (dwdy + dvdz);
  const fpdtype tzz = -2.0 * mu * dwdz - lambda * div;
  F[1] += txx * S[0] + txy * S[1] + txz * S[2];
  F[2] += txy * S[0] + tyy * S[1] + tyz * S[2];
  F[3] += txz * S[0] + tyz * S[1] + tzz * S[2];
  fpdtype dTx, dTy, dTz;
  facegrad(3, stateL.T, stateR.T, dTx, dTy, dTz);
  const fpdtype heatflux = -kappa * (dTx * S[0] + dTy * S[1] + dTz * S[2]);
  const fpdtype uf = 0.5 * (uR + uL), vf = 0.5 * (vR + vL), wf = 0.5 * (wR + wL);
  // f4 gathers the energy row and is written once at the end: the DF flux expands this unscoped and reads f4
  fpdtype f4 = (uf * txx + vf * txy + wf * txz) * S[0] + (uf * txy + vf * tyy + wf * tyz) * S[1] + (uf * txz + vf * tyz + wf * tzz) * S[2] + heatflux;
  fpdtype YlastL = 1.0, YlastR = 1.0;
  for (int n = 0; n < ${ns} - 1; n++) {
    YlastL -= QL[5 + n] * rhoinvL;
    YlastR -= QR[5 + n] * rhoinvR;
  }
  const auto YL = [&](const int n) { return n < ${ns} - 1 ? QL[5 + n] * rhoinvL : YlastL; };
  const auto YR = [&](const int n) { return n < ${ns} - 1 ? QR[5 + n] * rhoinvR : YlastR; };
  fpdtype Dk, Vc = 0.0, gradYns = 0.0, sumYh = 0.0, Yns = 1.0;
  const fpdtype rho = 0.5 * (QR[0] + QL[0]);
  for (int n = 0; n < ${ns} - 1; n++) {
    Dk = 0.5 * (trstateR[2 + n] + trstateL[2 + n]);
    fpdtype dYx, dYy, dYz;
    facegrad(4 + n, YL(n), YR(n), dYx, dYy, dYz);
    const fpdtype gradYk = dYx * S[0] + dYy * S[1] + dYz * S[2];
    gradYns -= gradYk;
    Vc += Dk * gradYk;
    const fpdtype Jk = -rho * Dk * gradYk;
    const fpdtype hk = 0.5 * (hspR[n] + hspL[n]);
    const fpdtype Yk = 0.5 * (YR(n) + YL(n));
    Yns -= Yk;
    sumYh += Yk * hk;
    F[5 + n] += Jk;
    f4 += Jk * hk;
  }
  Dk = 0.5 * (trstateR[2 + ${ns} - 1] + trstateL[2 + ${ns} - 1]);
  Vc += Dk * gradYns;
  // Vc keeps the species fluxes summing to zero and needs the sum over every species first, so each flux waits in F between the two passes
  const fpdtype corr = rho * Vc;
  for (int n = 0; n < ${ns} - 1; n++)
    F[5 + n] += 0.5 * (YR(n) + YL(n)) * corr;
  f4 += corr * sumYh;
  const fpdtype hns = 0.5 * (hspR[${ns} - 1] + hspL[${ns} - 1]);
  f4 += (-rho * Dk * gradYns + Yns * corr) * hns;
  F[4] += f4;
</%pg:macro>

<%pg:kernel name="flux" over="faces"
            S="in [3]"
            F="inout [${str(nflux)}]">
  ${pg.expand("alphadamping", "S, F")}
</%pg:kernel>
