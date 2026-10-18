// A real gas on a cubic equation of state (Soave-Redlich-Kwong), the ideal
// parts of cp and h from the thermally perfect polynomials plus the departures. A
// constant volume translation per species, mixed in mole fraction, shifts the
// volume the cubic gives, v = v_cubic - c: the density, the enthalpy by -p c and
// the derivatives follow; cp does not. Zt is the compressibility of the
// shifted volume, Z - c p / (Ru T).
#ifndef __realGas_H__
#define __realGas_H__

#include "conserved.hpp"
#include "cubic.hpp"
#include "kernel.hpp"
#include "species.hpp"

namespace realGas {

// the mixture state at one point, as the eos works it out
struct state {
  fpdtype p, T, rho, e, h, cp, gamma, c;
};

// what a cell keeps of the eos beyond p and T: gamma, cp, rho h, c, rho e,
// then each species' enthalpy, departure included, which no polynomial in T
// gives back
constexpr int qhComponents = 5 + ns;

// each species' enthalpy, as a callable of the species: read out of the
// cell's qh, where fromPrims and fromCons left them
template <class Qh>
KOKKOS_INLINE_FUNCTION auto enthalpies(const fpdtype, const Qh &qh) {
  return [=](const int n) { return qh(5 + n); };
}

// the cubic's form, v^2 + u b v + w b^2 = (v + sigma b)(v + eps b) in its
// denominator, sigma and eps that quadratic's roots and sq their spread;
// which cubic it is, and every constant of it, is the eos model's
// (eosModel.py), baked with the species data: the form here, the rest
// through each species' constants
constexpr fpdtype uRG = cubicU, wRG = cubicW, sigmaRG = cubicSigma,
                  epsRG = cubicEps, sqRG = cubicSigma - cubicEps,
                  invSqRG = 1.0 / sqRG;
constexpr fpdtype twoPiOver3 = 2.0943951023931954923;

// the cubic at T for the mixture X: its mixing coefficients and their
// temperature derivative, and, given p, the compressibility Z and dZ/dT
struct cubic {
  fpdtype am, bm, cm, dam, sumSqrtA, sumDa, Astar, Bstar, Cstar, Z, Zt, z1, z2,
      dZdT, dZtdT;
};

// the mixing coefficients that do not move with T, for the mole fractions
// X: b and c, and S2 = sum X cubicDa, which a's temperature derivatives take
KOKKOS_INLINE_FUNCTION void mixingConstants(const fpdtype *X, cubic &c) {
  c.bm = 0.0, c.cm = 0.0, c.sumDa = 0.0;
  for (int n = 0; n <= ns - 1; n++) {
    c.bm += X[n] * cubicB(n);
    c.cm += X[n] * volumeShift(n);
    c.sumDa += X[n] * cubicDa(n);
  }
}

// the attraction at T, with kij = 0, where the sums over the pairs are
// products of sums over the species: sqrt(a_i) = sqrt(a_c,i) |1 + m_i (1 -
// sqrt(T / Tc_i))|, the modulus being what sqrt(a_i a_j) of the pair sum
// takes, S1 = sum X sqrt(a_i), am = S1^2 and dam/dT = -S1 S2 / sqrt(T),
// since (da_i/dT) / sqrt(a_i) = -sqrt(a_c,i) m_i / sqrt(T Tc_i); sqrtAi
// receives each species' for the derivatives by composition
KOKKOS_INLINE_FUNCTION void attraction(const fpdtype T, const fpdtype *X,
                                       fpdtype *sqrtAi, cubic &c) {
  const fpdtype sqrtT = sqrt(T);
  fpdtype S1 = 0.0;
  for (int n = 0; n <= ns - 1; n++) {
    sqrtAi[n] = cubicSqrtAc(n) *
                fabs(1.0 + cubicM(n) * (1.0 - sqrtT * invSqrtTcrit(n)));
    S1 += X[n] * sqrtAi[n];
  }
  c.sumSqrtA = S1;
  c.am = S1 * S1;
  c.dam = -S1 * c.sumDa / sqrtT;
}

// both, at one T
KOKKOS_INLINE_FUNCTION void coefficients(const fpdtype T, const fpdtype *X,
                                         fpdtype *sqrtAi, cubic &c) {
  mixingConstants(X, c);
  attraction(T, X, sqrtAi, c);
}

// the cubic's dimensionless form at (p, T): A*, B*, C* and the two
// coefficients of its polynomial in Z that the derivatives by implicit
// differentiation need; gives p / (Ru T), which a caller that knows the
// volume turns into Z
KOKKOS_INLINE_FUNCTION fpdtype dimensionless(const fpdtype p, const fpdtype T,
                                             cubic &c) {
  const fpdtype invRuT = 1.0 / (Ru * T), pRT = p * invRuT;
  c.Astar = c.am * pRT * invRuT;
  c.Bstar = c.bm * pRT;
  c.Cstar = c.cm * pRT;
  const fpdtype B = c.Bstar, B2 = B * B;
  c.z1 = c.Astar + wRG * B2 - uRG * B - uRG * B2;
  c.z2 = -(1.0 + B - uRG * B);
  return pRT;
}

// Z of the cubic solved at (p, T), Cardano's roots, the stable phase's
KOKKOS_INLINE_FUNCTION void compressibility(const fpdtype p, const fpdtype T,
                                            cubic &c) {
  dimensionless(p, T, c);
  // https://www.e-education.psu.edu/png520/m11_p6.html
  const fpdtype B = c.Bstar, B2 = B * B, z1 = c.z1, z2 = c.z2;
  const fpdtype z0 = -(c.Astar * B + wRG * B2 * (1.0 + B));
  const fpdtype Q = (z2 * z2 - 3.0 * z1) * (1.0 / 9.0);
  const fpdtype R =
      (2.0 * z2 * z2 * z2 - 9.0 * z2 * z1 + 27.0 * z0) * (1.0 / 54.0);
  const fpdtype M = R * R - Q * Q * Q;
  const fpdtype z2o3 = z2 * (1.0 / 3.0);
  if (M > 0.0) {
    // one real root
    const fpdtype S = -copysign(cbrt(fabs(R) + sqrt(M)), R);
    c.Z = S + Q / S - z2o3;
  } else {
    // three, by the trigonometric form
    const fpdtype sqQ = sqrt(Q);
    const fpdtype theta3 = acos(R / (Q * sqQ)) * (1.0 / 3.0);
    const fpdtype x1 = -2.0 * sqQ * cos(theta3) - z2o3;
    const fpdtype x2 = -2.0 * sqQ * cos(theta3 + twoPiOver3) - z2o3;
    const fpdtype x3 = -2.0 * sqQ * cos(theta3 - twoPiOver3) - z2o3;
    c.Z = stableRoot(x1, x2, x3, c.Astar, c.Bstar, sigmaRG, epsRG);
  }
  c.Zt = c.Z - c.Cstar;
}

// dZ by implicit differentiation of the cubic, given dA* and dB* by the
// same variable
KOKKOS_INLINE_FUNCTION fpdtype dZ(const cubic &c, const fpdtype dAstar,
                                  const fpdtype dBstar) {
  const fpdtype B = c.Bstar, Z = c.Z, Z2 = Z * Z;
  const fpdtype dz0 =
      -(B * dAstar + (c.Astar + (2.0 * B + 3.0 * B * B) * wRG) * dBstar);
  const fpdtype dz1 = dAstar + (2.0 * B * (wRG - uRG) - uRG) * dBstar;
  const fpdtype dz2 = -(1.0 - uRG) * dBstar;
  return -(dz2 * Z2 + dz1 * Z + dz0) / (3.0 * Z2 + 2.0 * Z * c.z2 + c.z1);
}

// the temperature derivative of Z, and the departures of h and cp; the ideal
// parts are added by the caller
KOKKOS_INLINE_FUNCTION void departures(const fpdtype T, cubic &c,
                                       fpdtype &hDep, fpdtype &cpDep) {
  const fpdtype Tinv = 1.0 / T;
  const fpdtype damOverAm = c.dam / c.am;
  const fpdtype dAstardT = -2.0 * c.Astar * Tinv * (1.0 - 0.5 * T * damOverAm);
  const fpdtype dBstardT = -c.Bstar * Tinv;
  c.dZdT = dZ(c, dAstardT, dBstardT);
  c.dZtdT = c.dZdT + c.Cstar * Tinv;

  // in x = Z / B*: the cubic's denominator D = x^2 + u x + w, and the log
  // of its factors' ratio
  const fpdtype x = c.Z / c.Bstar, xm1 = x - 1.0;
  const fpdtype D = x * x + uRG * x + wRG;
  const fpdtype logRatio = log((x + epsRG) / (x + sigmaRG));
  const fpdtype invBmRu = 1.0 / (c.bm * Ru);
  const fpdtype hRes = invBmRu * invSqRG * (c.am * Tinv - c.dam) * logRatio;
  const fpdtype qd = c.dam * invBmRu, qa = c.am * invBmRu * Tinv;
  const fpdtype num = D - qd * xm1;
  cpDep = hRes * 0.5 * T * damOverAm +
          num * num / (D * D - qa * (2.0 * x + uRG) * xm1 * xm1) - 1.0;
  hDep = hRes + (c.Z - 1.0) - c.Cstar;
}

// one species' ideal cp / R and h / (R T) at ln T and 1 / T, Horner over
// the polynomials
KOKKOS_INLINE_FUNCTION void ideal(const int n, const fpdtype u,
                                  const fpdtype Tinv, fpdtype &cpR,
                                  fpdtype &hRT) {
  cpR = 0.0, hRT = 0.0;
  for (int m = cpPolyTerms - 1; m >= 0; m--)
    cpR = cpR * u + cpPoly(n, m);
  for (int m = hPolyTerms - 1; m >= 0; m--)
    hRT = hRT * u + hPoly(n, m);
  hRT += hRef(n) * Tinv;
}

// the mixture's ideal internal energy per mass and its temperature
// derivative, the ideal cv
template <class Yf>
KOKKOS_INLINE_FUNCTION void idealEnergy(const fpdtype T, const Yf &Y,
                                        fpdtype &e, fpdtype &cv) {
  const fpdtype u = log(T), Tinv = 1.0 / T;
  e = 0.0, cv = 0.0;
  for (int n = 0; n <= ns - 1; n++) {
    fpdtype cpR, hRT;
    ideal(n, u, Tinv, cpR, hRT);
    const fpdtype Rn = Ru * MWinv(n);
    e += Y(n) * Rn * T * (hRT - 1.0);
    cv += Y(n) * Rn * (cpR - 1.0);
  }
}

// the mixture h and cp at T, departures first, and each species' partial
// enthalpy into hi[]: its ideal enthalpy plus its partial molar residual
// enthalpy over its molar mass, -Ru T^2 (d ln phi_i / dT) at fixed p and
// composition, the fugacity coefficient of a cubic with the van der Waals
// rules being ln phi_i = (b_i/b)(Z - 1) - ln(Z - B) - qbar_i I with
// q = a/(b Ru T), qbar_i = q (2 sum_j x_j a_ij / a - b_i/b) and
// I = ln((Z + sigma B)/(Z + eps B)) / (sigma - eps); a species' volume
// translation adds -p c_i. The partial enthalpies sum, in mass fraction,
// to the mixture's.
template <class Yf>
KOKKOS_INLINE_FUNCTION void properties(const fpdtype T, const Yf &Y,
                                       const fpdtype MWmix,
                                       const fpdtype *sqrtAi, const cubic &c,
                                       const fpdtype hDep, const fpdtype cpDep,
                                       fpdtype &h, fpdtype &cp, fpdtype *hi) {
  const fpdtype invMWmix = 1.0 / MWmix;
  h = Ru * T * hDep * invMWmix;
  cp = Ru * cpDep * invMWmix;
  const fpdtype u = log(T), Tinv = 1.0 / T, sqrtT = sqrt(T);

  // the temperature derivative of ln phi_i at fixed p and x, in the pieces
  // every species shares: Z, B = Bstar and their derivatives, I and its,
  // q and its
  const fpdtype Z = c.Z, B = c.Bstar, ZT = c.dZdT, BT = -B * Tinv;
  const fpdtype invZsB = 1.0 / (Z + sigmaRG * B), invZeB = 1.0 / (Z + epsRG * B);
  const fpdtype I = log((Z + sigmaRG * B) * invZeB) * invSqRG;
  const fpdtype IT =
      ((ZT + sigmaRG * BT) * invZsB - (ZT + epsRG * BT) * invZeB) * invSqRG;
  const fpdtype q = c.Astar / B, qT = q * (c.dam / c.am - Tinv);
  const fpdtype dlnZmB = (ZT - BT) / (Z - B);
  const fpdtype invS1 = 1.0 / c.sumSqrtA, invBm = 1.0 / c.bm;
  const fpdtype rTscale = invS1 * invS1 / sqrtT;
  // -p c_i in the dimensionless form Bstar / bm = p / (Ru T)
  const fpdtype shift = B * invBm * Ru * T;
  const fpdtype minusRuT2 = -Ru * T * T;

  for (int n = 0; n <= ns - 1; n++) {
    fpdtype cpR, hRT;
    ideal(n, u, Tinv, cpR, hRT);
    const fpdtype Rn = Ru * MWinv(n);
    hi[n] = hRT * T * Rn;
    cp += cpR * Rn * Y(n);
    h += hi[n] * Y(n);
    // the partial molar residual enthalpy: 2 sum_j x_j a_nj / a is
    // 2 sqrt(a_n) / sum_j x_j sqrt(a_j) under kij = 0, its temperature
    // derivative from the species' cubicDa
    const fpdtype r = 2.0 * sqrtAi[n] * invS1;
    const fpdtype rT =
        (sqrtAi[n] * c.sumDa - cubicDa(n) * c.sumSqrtA) * rTscale;
    const fpdtype bb = cubicB(n) * invBm;
    const fpdtype qbar = q * (r - bb), qbarT = qT * (r - bb) + q * rT;
    const fpdtype dlnphidT = bb * ZT - dlnZmB - qbarT * I - qbar * IT;
    hi[n] += (minusRuT2 * dlnphidT - volumeShift(n) * shift) * MWinv(n);
  }
}

// gamma and the speed of sound from Z's pressure derivative
KOKKOS_INLINE_FUNCTION void ratios(const fpdtype p, const fpdtype T,
                                   const fpdtype rho, const fpdtype cp,
                                   const cubic &c, fpdtype &gamma,
                                   fpdtype &cs) {
  const fpdtype invp = 1.0 / p, Tinv = 1.0 / T, invZt = 1.0 / c.Zt;
  const fpdtype dZtdp = dZ(c, c.Astar * invp, c.Bstar * invp) - c.Cstar * invp;
  const fpdtype drhodp = rho * invp * (1.0 - p * dZtdp * invZt);
  // drho/dT over rho
  const fpdtype r = -Tinv * (1.0 + T * c.dZtdT * invZt);
  gamma = drhodp / (drhodp - (T / cp) * r * r);
  cs = sqrt(fabs(gamma / drhodp));
}

// the state from pressure, temperature and mass fractions; hi(n, value)
// receives each species' enthalpy, departure included
template <class Yf, class Hf>
KOKKOS_INLINE_FUNCTION state fromPrims(const fpdtype p, const fpdtype T,
                                       const Yf &Y, const Hf &hi) {
  state s;
  fpdtype X[ns], sqrtAi[ns], his[ns];
  const fpdtype MWmix = moleFractions(Y, X);
  const fpdtype Rmix = Ru / MWmix;
  cubic c;
  coefficients(T, X, sqrtAi, c);
  compressibility(p, T, c);
  fpdtype hDep, cpDep;
  departures(T, c, hDep, cpDep);
  properties(T, Y, MWmix, sqrtAi, c, hDep, cpDep, s.h, s.cp, his);
  s.p = p;
  s.T = T;
  // p / rho is Zt Rmix T
  const fpdtype pOverRho = c.Zt * Rmix * T;
  s.rho = p / pOverRho;
  s.e = s.h - pOverRho;
  ratios(p, T, s.rho, s.cp, c, s.gamma, s.c);
  for (int n = 0; n <= ns - 1; n++) {
    hi(n, his[n]);
  }
  return s;
}

// the state from density, internal energy and mass fractions: Newton on T
// at the cell's volume, a fixed count of steps as tpg takes -- every thread
// of a warp on the same path -- then the state at that T once. The Newton
// is quadratic: from the last state's T, off by a fraction of a kelvin
// over a stage, two steps pass the enthalpy's rounding floor of 3e-12 of
// T, and three do from ten kelvin off.
constexpr int newtonSteps = 3;
template <class Yf, class Hf>
KOKKOS_INLINE_FUNCTION state fromCons(const fpdtype rho, const fpdtype e,
                                      const Yf &Y, const fpdtype Tguess,
                                      const Hf &hi) {
  state s;
  fpdtype X[ns], sqrtAi[ns], his[ns];
  const fpdtype MWmix = moleFractions(Y, X);
  const fpdtype invMWmix = 1.0 / MWmix;
  // molar volume, the shifted one the state holds, and the cubic's own
  const fpdtype Vm = MWmix / rho;
  cubic c;
  mixingConstants(X, c);
  const fpdtype v = Vm + c.cm;
  const fpdtype vs = v + sigmaRG * c.bm, ve = v + epsRG * c.bm;
  // The residual is e(T, V) - e at the cell's volume: the ideal energy of
  // the species polynomials plus the cubic's residual energy,
  // -(a - T a') I(V) with I(V) = ln((V + sigma b) / (V + eps b)) / (b sq),
  // and its exact derivative is cv, the ideal cv plus T a'' I(V), where
  // a'' = (S2^2 - a') / (2 T) from the two sums the coefficients keep.
  const fpdtype I = log(vs / ve) * invSqRG / c.bm;
  fpdtype T = (Tguess < 1.0) ? 300.0 : Tguess;
  for (int k = 0; k < newtonSteps; k++) {
    attraction(T, X, sqrtAi, c);
    const fpdtype Tinv = 1.0 / T;
    const fpdtype app = 0.5 * (c.sumDa * c.sumDa - c.dam) * Tinv;
    fpdtype eIdeal, cvIdeal;
    idealEnergy(T, Y, eIdeal, cvIdeal);
    const fpdtype f = eIdeal - (c.am - T * c.dam) * I * invMWmix - e;
    const fpdtype cv = cvIdeal + T * app * I * invMWmix;
    T -= f / cv;
  }
  // the state at T: p from the volume, and Z is the volume's, no root to
  // find
  attraction(T, X, sqrtAi, c);
  const fpdtype p = Ru * T / (v - c.bm) - c.am / (vs * ve);
  c.Z = dimensionless(p, T, c) * v;
  c.Zt = c.Z - c.Cstar;
  fpdtype hDep, cpDep;
  departures(T, c, hDep, cpDep);
  properties(T, Y, MWmix, sqrtAi, c, hDep, cpDep, s.h, s.cp, his);
  s.rho = rho;
  s.e = e;
  s.p = p;
  s.T = T;
  ratios(p, T, rho, s.cp, c, s.gamma, s.c);
  for (int n = 0; n <= ns - 1; n++) {
    hi(n, his[n]);
  }
  return s;
}

// The energy's slopes the double flux freezes, at (rho, T, Y) with e the
// energy per mass there: theta = d(rho e)/dp at fixed density, rho cv over
// dp/dT at the cell's volume, cv the ideal cv plus the cubic's T a'' I(V) as
// the energy Newton has it; and eps = d(rho e)/drho at fixed pressure, e -
// (T dp/dT - p) / rho - theta (dp/drho)_T, the cubic's derivatives closed,
// (dp/drho)_T from its (dp/dv)_T over the molar volume MWmix / rho.
template <class Yf>
KOKKOS_INLINE_FUNCTION void energySlopes(const fpdtype rho, const fpdtype T,
                                         const fpdtype e, const Yf &Y,
                                         fpdtype &theta, fpdtype &eps) {
  fpdtype X[ns], sqrtAi[ns];
  const fpdtype MWmix = moleFractions(Y, X);
  const fpdtype Vm = MWmix / rho;
  cubic c;
  coefficients(T, X, sqrtAi, c);
  const fpdtype v = Vm + c.cm;
  const fpdtype vs = v + sigmaRG * c.bm, ve = v + epsRG * c.bm;
  const fpdtype invVmb = 1.0 / (v - c.bm), invD = 1.0 / (vs * ve);
  const fpdtype I = log(vs / ve) * invSqRG / c.bm;
  const fpdtype app = 0.5 * (c.sumDa * c.sumDa - c.dam) / T;
  fpdtype eIdeal, cvIdeal;
  idealEnergy(T, Y, eIdeal, cvIdeal);
  const fpdtype cv = cvIdeal + T * app * I / MWmix;
  const fpdtype p = Ru * T * invVmb - c.am * invD;
  const fpdtype dpdT = Ru * invVmb - c.dam * invD;
  const fpdtype dpdrho = MWmix / (rho * rho) *
                         (Ru * T * invVmb * invVmb - c.am * (vs + ve) * invD * invD);
  theta = rho * cv / dpdT;
  eps = e - (T * dpdT - p) / rho - theta * dpdrho;
}

// the state from density, pressure and mass fractions, what the double flux
// asks: Newton on T at the cell's volume for the cubic's pressure, whose
// derivative there is closed, dp/dT = Ru / (v - b) - a' / D, the same fixed
// count of steps, then the state at that T once
template <class Yf, class Hf>
KOKKOS_INLINE_FUNCTION state fromRhoP(const fpdtype rho, const fpdtype p,
                                      const Yf &Y, const fpdtype Tguess,
                                      const Hf &hi) {
  state s;
  fpdtype X[ns], sqrtAi[ns], his[ns];
  const fpdtype MWmix = moleFractions(Y, X);
  const fpdtype Vm = MWmix / rho;
  cubic c;
  mixingConstants(X, c);
  const fpdtype v = Vm + c.cm;
  const fpdtype invVmb = 1.0 / (v - c.bm);
  const fpdtype invD = 1.0 / ((v + sigmaRG * c.bm) * (v + epsRG * c.bm));
  fpdtype T = (Tguess < 1.0) ? 300.0 : Tguess;
  for (int k = 0; k < newtonSteps; k++) {
    attraction(T, X, sqrtAi, c);
    const fpdtype f = Ru * T * invVmb - c.am * invD - p;
    const fpdtype dpdT = Ru * invVmb - c.dam * invD;
    T -= f / dpdT;
  }
  attraction(T, X, sqrtAi, c);
  c.Z = dimensionless(p, T, c) * v;
  c.Zt = c.Z - c.Cstar;
  fpdtype hDep, cpDep;
  departures(T, c, hDep, cpDep);
  properties(T, Y, MWmix, sqrtAi, c, hDep, cpDep, s.h, s.cp, his);
  s.rho = rho;
  s.p = p;
  s.T = T;
  s.e = s.h - p / rho;
  ratios(p, T, rho, s.cp, c, s.gamma, s.c);
  for (int n = 0; n <= ns - 1; n++) {
    hi(n, his[n]);
  }
  return s;
}

// the density's derivatives at (p, T, Y) for the dual time preconditioning:
// rho = p MWmix / (Zt Ru T), so each is rho times the log derivative, Z's by
// the implicit differentiation above. By p and T as ratios has them; by
// each of the first ns - 1 mass fractions with the last taking up the
// change, through the mixture's molecular weight and the mixing rules for
// the cubic's coefficients.
template <class Yf>
KOKKOS_INLINE_FUNCTION void densityDerivatives(const fpdtype p, const fpdtype T,
                                               const fpdtype rho, const Yf &Y,
                                               fpdtype &rho_p, fpdtype &rho_T,
                                               fpdtype *rho_Y) {
  fpdtype X[ns], sqrtAi[ns];
  const fpdtype MWmix = moleFractions(Y, X);
  const fpdtype invMWmix = 1.0 / MWmix;
  cubic c;
  coefficients(T, X, sqrtAi, c);
  compressibility(p, T, c);
  fpdtype hDep, cpDep;
  departures(T, c, hDep, cpDep);
  const fpdtype invp = 1.0 / p, Tinv = 1.0 / T, invZt = 1.0 / c.Zt;
  rho_p = rho * invp *
          (1.0 - (p * dZ(c, c.Astar * invp, c.Bstar * invp) - c.Cstar) * invZt);
  rho_T = -rho * Tinv * (1.0 + T * c.dZtdT * invZt);
  // d am / d X_k of am = (sum X sqrt(a))^2
  const fpdtype twoSumSqrtA = 2.0 * c.sumSqrtA;
  const fpdtype invRuT = Tinv / Ru, pRT = p * invRuT;
  for (int n = 0; n < ns - 1; n++) {
    // the mixture's molecular weight, 1 / sum(Y / MW), by Y_n less Y_last,
    // as a log derivative
    const fpdtype dlnMWmix = -MWmix * (MWinv(n) - MWinv(ns - 1));
    // the mole fractions follow: X_k = Y_k MWmix / MW_k
    fpdtype dam = 0.0, dbm = 0.0, dcm = 0.0;
    for (int k = 0; k <= ns - 1; k++) {
      fpdtype dX = X[k] * dlnMWmix;
      if (k == n)
        dX += MWmix * MWinv(n);
      if (k == ns - 1)
        dX -= MWmix * MWinv(ns - 1);
      dam += twoSumSqrtA * sqrtAi[k] * dX;
      dbm += cubicB(k) * dX;
      dcm += volumeShift(k) * dX;
    }
    const fpdtype dZtdY = dZ(c, dam * pRT * invRuT, dbm * pRT) - dcm * pRT;
    rho_Y[n] = rho * (dlnMWmix - dZtdY * invZt);
  }
}

} // namespace realGas

#endif
