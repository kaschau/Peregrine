#include "conserved.hpp"
#include "diffusion.hpp"
#include "dilute.hpp"
#include "kernel.hpp"
#include "thermo/realGas.hpp"

// Dense-fluid transport by residual entropy scaling: kinetic theory's dilute
// mixture, mixed by the case's rule, plus a residual that is a function of
// the cubic's residual entropy alone. Made dimensionless by the number
// density and the thermal velocity, and divided by s+^(2/3), the residual
// viscosity and conductivity of a fluid each follow one curve in
// s+ = -s_res / R, its shape set by three coefficients for the viscosity
// and four for the conductivity, which mix in mole fraction. The effective
// molecular mass is the mole-fraction mean for the viscosity and the
// mass-fraction mean of M^(-1/2) for the conductivity. The residual entropy
// is the case's own cubic's, from the mixing coefficients the eos forms, at
// the cubic's volume; the species' coefficients were fitted on
// Soave-Redlich-Kwong with kij = 0. No critical enhancement: it matters
// within a fifth of Tc near pc and would take four more constants a species.
//
// Probing the link between residual entropy and viscosity of molecular fluids
//     Ian H. Bell
//     PNAS 2019, 116, 4070-4079
// Linking viscosity to equations of state using residual entropy scaling theory
//     Yang, Xiao, Thol, Richter, Bell
//     Int. J. Thermophys. 2022, 43, 183
// Entropy scaling of viscosity IV: application to 124 industrially important fluids
//     Martinek et al.
//     J. Chem. Eng. Data 2025, 70, 727-742
// Linking thermal conductivity to equations of state using the residual entropy
// scaling theory
//     Li, Duan, Yang
//     Ind. Eng. Chem. Res. 2024, 63, 18160-18175
// Viscosity and thermal conductivity models of 151 common fluids based on
// residual entropy scaling and cubic equations of state
//     Yang et al.
//     ACS Omega 2025, 10, 6124-6140

PG_RANGE(cellCenters)
struct entropyScaling {
  cellVecIn Q, q, qh;
  cellVecOut qt;
  // the prefactors of the two residuals with N_A per kmol and kB in J/K:
  // rhoN^(2/3) sqrt(m kB T) = cMu rho^(2/3) MWmix^(-1/6) sqrt(T) for the
  // viscosity, cMu = N_A^(1/6) kB^(1/2); rhoN^(2/3) kB sqrt(kB T / m) =
  // cKappa rho^(2/3) MWmix^(-2/3) sqrt(T) sum(Y_i MW_i^(-1/2)) for the
  // conductivity, cKappa = N_A^(7/6) kB^(3/2)
  static constexpr fpdtype cMu = 1.079775473285991e-07,
                           cKappa = 8.97775280863514e-04;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const fpdtype &T = q(1);
    const fpdtype lnT = log(T);
    const fpdtype sqrtT = sqrt(T);
    fpdtype X[ns];

    // the mole fractions off the conserved state, the last mass fraction
    // kept non-negative
    const fpdtype rhoinv = 1.0 / Q(0);
    const auto Yof = massFractions(Q, rhoinv);
    const auto Y = [&](const int n) {
      return n == ns - 1 ? fmax(0.0, Yof(n)) : Yof(n);
    };
    const fpdtype MWmix = moleFractions(Y, X);
    const fpdtype invMWmix = 1.0 / MWmix;

    // the dilute mixture
    fpdtype muDilute, kappaDilute;
    diluteTransport(T, lnT, X, muDilute, kappaDilute);

    // the mixture's coefficients, the conductivity's mass and the cubic's
    // mixing coefficients, in one pass over the species
    fpdtype nMu[3] = {0.0, 0.0, 0.0}, nKappa[4] = {0.0, 0.0, 0.0, 0.0};
    fpdtype invSqrtMass = 0.0;
    for (int n = 0; n <= ns - 1; n++) {
      for (int k = 0; k < 3; k++)
        nMu[k] += X[n] * resMu(n, k);
      for (int k = 0; k < 4; k++)
        nKappa[k] += X[n] * resKappa(n, k);
      invSqrtMass += Y(n) * invSqrtMW(n);
    }
    fpdtype sqrtAi[ns];
    realGas::cubic c;
    realGas::coefficients(T, X, sqrtAi, c);

    // the cubic's residual entropy at its own volume, over -R:
    // s+ = -[ln(1 - b/v) + (da/dT) / (b R) ln(1 + b/v)], positive
    const fpdtype bOverV = c.bm / (MWmix * rhoinv + c.cm);
    const fpdtype splus =
        fmax(-(log(1.0 - bOverV) + c.dam / (c.bm * Ru) * log(1.0 + bOverV)),
             1.0e-12);
    const fpdtype cbrtS = cbrt(splus);
    const fpdtype invS23 = 1.0 / (cbrtS * cbrtS);
    const fpdtype cbrtRho = cbrt(Q(0));
    const fpdtype rho23 = cbrtRho * cbrtRho;
    const fpdtype cbrtInvMW = cbrt(invMWmix);

    // viscosity: ln(eta+ + 1) in s+^1.8, s+^2.4, s+^2.8, the powers from
    // one fifth root
    const fpdtype u = pow(splus, 0.2);
    const fpdtype u3 = u * u * u;
    const fpdtype u9 = u3 * u3 * u3;
    const fpdtype u12 = u9 * u3;
    const fpdtype u14 = u12 * u * u;
    const fpdtype etaPlus = exp(nMu[0] * u9 + nMu[1] * u12 + nMu[2] * u14) - 1.0;
    qt(0) = muDilute + etaPlus * invS23 * cMu * rho23 *
                           sqrt(cbrtInvMW) * sqrtT;

    // thermal conductivity: lambda+ in s+, s+^1.5, s+^2, s+^2.5
    const fpdtype r = sqrt(splus);
    const fpdtype lambdaPlus = nKappa[0] * splus + nKappa[1] * splus * r +
                               nKappa[2] * splus * splus +
                               nKappa[3] * splus * splus * r;
    const fpdtype kappa = kappaDilute + lambdaPlus * invS23 *
                                            cKappa * rho23 *
                                            cbrtInvMW * cbrtInvMW * sqrtT *
                                            invSqrtMass;
    qt(1) = kappa;

    // the species diffusion coefficients from the case's diffusion model
    fpdtype D[ns];
    diffusion::coefficients({q(0), T, lnT, rhoinv, qh(1), kappa, MWmix, X}, D);
    for (int n = 0; n <= ns - 1; n++) {
      qt(2 + n) = D[n];
    }
  }
};

PG_ABI void pgEntropyScaling(const entropyScaling &k, const pgTiling &t) {
  forCells("Entropy scaling trans props", t, k);
}
