#include "conserved.hpp"
#include "diffusion.hpp"
#include "dilute.hpp"
#include "kernel.hpp"

// Dense-fluid transport: kinetic theory's dilute mixture, mixed by the
// case's rule, times Chung's density enhancement. The enhancement is the
// ratio of Chung's dense to dilute value for the mixture taken as one
// nonpolar fluid, whose collision diameter, well depth, weight and acentric
// factor come from every species' by Chung's quadratic mixing rules. The
// polar physics stays in the dilute part, where kinetic theory's
// Stockmayer collision integrals have it: Chung's own dipole terms are not
// applied, because under the one-fluid rule a few mole percent of water
// would polarize the whole fluid's dense coefficients and swing its
// viscosity by tens of percent. Written in SI as Poling states Chung, the
// critical volume in cm^3/mol as its constants want it.
//
// Generalized Multiparameter Correlation for Nonpolar and Polar Fluid Transport
// Properties
//     Ting-Horng Chung, Ajlan, Lee, Starling
//     Ind. Eng. Chem. Res. 1988, 27, 671-679
//
// The Properties of Gases and Liquids, 5th Edition, 2001
//     Poling, Prausnitz, O'Connell: 9-5, 9-6 and 10-5

PG_RANGE(cellCenters)
struct chungDenseGas {
  cellVecIn Q, q, qh;
  cellVecOut qt;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const fpdtype &T = q(1);
    const fpdtype lnT = log(T);
    fpdtype X[ns];

    // the mole fractions off the conserved state, the last mass fraction
    // kept non-negative
    const fpdtype rhoinv = 1.0 / Q(0);
    const auto Yof = massFractions(Q, rhoinv);
    const auto Y = [&](const int n) {
      return n == ns - 1 ? fmax(0.0, Yof(n)) : Yof(n);
    };
    const fpdtype MWmix = moleFractions(Y, X);

    // the dilute mixture
    fpdtype muDilute, kappaDilute;
    diluteTransport(T, lnT, X, muDilute, kappaDilute);

    // the one fluid: sigma_m^3, (eps/k)_m, M_m and omega_m over the pairs,
    // sigma_ij = sqrt(sigma_i sigma_j), eps_ij = sqrt(eps_i eps_j), omega_ij
    // the mean, M_ij the reduced mass times two; and its ideal-gas cv,
    // cp/R - 1 of each species by its polynomial, which Psi takes
    fpdtype sigma3 = 0.0, epsSigma3 = 0.0, massSum = 0.0, omegaSigma3 = 0.0,
            cvR = 0.0;
    for (int i = 0; i <= ns - 1; i++) {
      fpdtype cpR = 0.0;
      for (int m = cpPolyTerms - 1; m >= 0; m--) {
        cpR = cpR * lnT + cpPoly(i, m);
      }
      cvR += X[i] * (cpR - 1.0);
      for (int j = 0; j <= ns - 1; j++) {
        const fpdtype xx = X[i] * X[j];
        const fpdtype sigma2ij = chungSigma(i) * chungSigma(j);
        const fpdtype sigma3ij = sigma2ij * sqrt(sigma2ij);
        const fpdtype epsij = sqrt(chungEpsK(i) * chungEpsK(j));
        const fpdtype Mij = 2.0 * MW(i) * MW(j) / (MW(i) + MW(j));
        sigma3 += xx * sigma3ij;
        epsSigma3 += xx * epsij * sigma3ij;
        massSum += xx * epsij * sigma2ij * sqrt(Mij);
        omegaSigma3 += xx * 0.5 * (acentric(i) + acentric(j)) * sigma3ij;
      }
    }
    const fpdtype epsK = epsSigma3 / sigma3;
    const fpdtype sigma2 = pow(sigma3, 2.0 / 3.0);
    const fpdtype Mm = pow(massSum / (epsK * sigma2), 2.0);
    const fpdtype omega = omegaSigma3 / sigma3;
    // its critical point, in Chung's units
    const fpdtype Tc = 1.2593 * epsK;
    const fpdtype Vc = sigma3 / (0.809 * 0.809 * 0.809);
    const fpdtype Fc = 1.0 - 0.2756 * omega;
    const fpdtype Vc23 = pow(Vc, 2.0 / 3.0);
    const fpdtype Mprime = Mm * 1.0e-3;

    // the correlation's coefficients at this fluid's omega
    fpdtype E[10], B[7];
    for (int i = 0; i < 10; i++) {
      E[i] = chungE(i, 0) + chungE(i, 1) * omega;
    }
    for (int i = 0; i < 7; i++) {
      B[i] = chungB(i, 0) + chungB(i, 1) * omega;
    }

    const fpdtype Tr = T / Tc;
    const fpdtype Tstar = T / epsK;
    const fpdtype Omegav = 1.16145 * pow(Tstar, -0.14874) +
                           0.52487 * exp(-0.77320 * Tstar) +
                           2.16178 * exp(-2.43787 * Tstar);
    // the reduced density: rho in mol/cm^3 times Vc over six
    const fpdtype Yy = Q(0) / Mm * 1.0e-3 * Vc / 6.0;
    const fpdtype G1 = (1.0 - 0.5 * Yy) / pow(1.0 - Yy, 3.0);

    // viscosity: Chung's dense eta* over its dilute eta*, the latter
    // sqrt(T*) Fc / Omegav
    const fpdtype G2 =
        (E[0] * (1.0 - exp(-E[3] * Yy)) / Yy + E[1] * G1 * exp(E[4] * Yy) +
         E[2] * G1) /
        (E[0] * E[3] + E[1] + E[2]);
    const fpdtype etaStarStar = E[6] * pow(Yy, 2.0) * G2 *
                                exp(E[7] + E[8] / Tstar + E[9] * pow(Tstar, -2.0));
    const fpdtype muRatio =
        (1.0 / G2 + E[5] * Yy) + etaStarStar * Omegav / (sqrt(Tstar) * Fc);
    qt(0) = muDilute * muRatio;

    // thermal conductivity: Chung's dense lambda over its dilute
    // 31.2 eta0 Psi / M', eta0 in Pa s
    const fpdtype eta0 = 4.0785e-6 * sqrt(Mm * T) / (Vc23 * Omegav) * Fc;
    const fpdtype alpha = cvR - 1.5;
    const fpdtype beta = 0.7862 - 0.7109 * omega + 1.3168 * omega * omega;
    const fpdtype Z = 2.0 + 10.5 * Tr * Tr;
    const fpdtype Psi =
        1.0 + alpha * ((0.215 + 0.28288 * alpha - 1.061 * beta + 0.26665 * Z) /
                       (0.6366 + beta * Z + 1.061 * alpha * beta));
    const fpdtype lambda0 = 31.2 * eta0 * Psi / Mprime;
    const fpdtype H2 =
        (B[0] * (1.0 - exp(-B[3] * Yy)) / Yy + B[1] * G1 * exp(B[4] * Yy) +
         B[2] * G1) /
        (B[0] * B[3] + B[1] + B[2]);
    const fpdtype kappaRatio =
        (1.0 / H2 + B[5] * Yy) +
        3.586e-3 * sqrt(Tc / Mprime) / Vc23 * B[6] * pow(Yy, 2.0) * H2 * sqrt(Tr) /
            lambda0;
    const fpdtype kappa = kappaDilute * kappaRatio;
    qt(1) = kappa;

    // the species diffusion coefficients from the case's diffusion model
    fpdtype D[ns];
    diffusion::coefficients({q(0), T, lnT, rhoinv, qh(1), kappa, MWmix, X}, D);
    for (int n = 0; n <= ns - 1; n++) {
      qt(2 + n) = D[n];
    }
  }
};

PG_ABI void pgChungDenseGas(const chungDenseGas &k, const pgTiling &t) {
  forCells("Chung dense gas trans props", t, k);
}
