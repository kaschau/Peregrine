// The double flux of Ma, Lv and Ihme (J. Comput. Phys. 340, 2017), after
// Abgrall and Karni: at the start of a step every cell freezes a linear
// relation between its internal energy and its pressure, density and
// partial densities,
//
//   rho e = theta p + eps rho + sum_n zeta_n rho Y_n + kappa,
//
// through its own state and advances its energy in that relation for the
// step. The coefficients are the eos's slopes at the state: theta is
// d(rho e)/dp at fixed density and composition, zeta_n is d(rho e)/d(rho
// Y_n) at fixed density and pressure with the last species giving way, eps
// is d(rho e)/drho at fixed pressure and fixed partial densities of the
// carried species, and kappa makes the relation exact at the state. It is
// the tangent plane of the energy in every variable the fluxes and the
// chemistry move, so the frozen system has the eos's speed of sound and its
// pressure response to heating, mixing and reaction, and what the relation
// leaves off the eos's energy after a step is second order in the state's
// change. The energy flux is written twice at every face, once in each
// cell's relation -- the double flux: a formula forms its energy flux with
// the other side's energy re-expressed in this side's relation, into F(4)
// for the face's left cell and F(ne) for its right, and applyFlux takes
// each cell's own; a diffusive flux, the same for both, is added to both.
// The pressure after a stage is the relation's, the eos giving the rest at
// that density and pressure (stateFromRelation). Pressure equilibrium
// across a contact between two fluids then survives the update, at the
// cost of the energy: the two cells at a face take different energy
// fluxes, and the next freeze resets each cell's to the eos's. The
// coefficients sit in qh past what the eos keeps,
// theta, eps, kappa, then zeta_n for the ns - 1 carried species; the freeze
// runs over the halos too, whose state is their neighbors' since the last
// exchange, so a halo's relation is its neighbor's.
//
// Theta is the tangent in the pressure, not Ma et al.'s 1 / (gamma* - 1)
// with gamma* = rho c^2 / p: a state re-made from its neighbor's energy
// through a copy of the relation sits a rounding off it, and a re-freeze
// moves its pressure by (1 - theta_tangent / theta) of the offset each
// step, tenfold for a dense fluid under the speed-of-sound form, nothing
// under the tangent. The slope in the density is what gives the speed of
// sound back, which a relation in the pressure alone puts at sqrt((1 +
// theta) p / (rho theta)), a third of the eos's in a liquid; the slopes in
// the composition are what let the heat of reaction, and the formation
// enthalpy species diffusion carries, reach the pressure -- without them a
// closed box does not ignite and a diffusing layer heats spuriously. Kappa
// is taken as rho (e - eps_Y) - theta p, eps_Y the density slope at fixed
// composition, so theta p + kappa is rho (e - eps_Y), positive, and the
// harmonic mean the apply takes of it over the two sides is safe.
//
// An integrator that differences the state against earlier states keeps
// them in the state's variables, so at each freeze they are taken through
// their relation's pressure into the new one (relationPressure ahead of the
// freeze, relationEnergy after), and the pseudo system's Jacobian is the
// relation's (invertDQ), in the pressure and the scaled temperature
// p / (rho R*): the density by those two alone, the energy without a
// temperature column, the zeta_n for species enthalpy differences, and
// the preconditioner standing in for the density's pressure slope on the
// conserved state over the density, as Weiss and Smith have it.
#ifndef __doubleFlux_H__
#define __doubleFlux_H__

#include "thermo/eos.hpp"

namespace doubleFlux {

// where the coefficients sit in qh, and how many there are
constexpr int slotTheta = eos::qhComponents, slotEps = slotTheta + 1,
              slotKappa = slotTheta + 2, slotZeta = slotTheta + 3;
constexpr int slots = ns + 2;

// the kinetic energy per volume of a conserved vector
template <class V> KOKKOS_INLINE_FUNCTION fpdtype kinetic(const V &Q) {
  return 0.5 * (Q(1) * Q(1) + Q(2) * Q(2) + Q(3) * Q(3)) / Q(0);
}

// the relation's composition part of a conserved vector, sum zeta_n rho Y_n
template <class V, class Qh>
KOKKOS_INLINE_FUNCTION fpdtype composition(const V &Q, const Qh &qh) {
  fpdtype s = 0.0;
  for (int n = 0; n < ns - 1; n++)
    s += qh(slotZeta + n) * Q(5 + n);
  return s;
}

// the relation's pressure of a conserved vector
template <class V, class Qh>
KOKKOS_INLINE_FUNCTION fpdtype pressure(const V &Q, const Qh &qh) {
  return (Q(4) - kinetic(Q) - qh(slotEps) * Q(0) - composition(Q, qh) -
          qh(slotKappa)) /
         qh(slotTheta);
}

// the conserved energy of a pressure, for a conserved vector's density,
// momentum and species
template <class V, class Qh>
KOKKOS_INLINE_FUNCTION fpdtype energy(const fpdtype p, const V &Q,
                                      const Qh &qh) {
  return qh(slotTheta) * p + qh(slotEps) * Q(0) + composition(Q, qh) +
         qh(slotKappa) + kinetic(Q);
}

// the part of the energy per mass that varies with pressure and density,
// (theta p + kappa) / rho; eps and the composition part are the rest
template <class Qh>
KOKKOS_INLINE_FUNCTION fpdtype thermal(const fpdtype p, const fpdtype rhoinv,
                                       const Qh &qh) {
  return (qh(slotTheta) * p + qh(slotKappa)) * rhoinv;
}

// the energy per volume one side's face state has in the relation of qh: the
// relation over the state's density, pressure and partial densities, plus
// its kinetic energy. A formula takes both sides this way, in each cell's
// relation: a reconstructed state's eos energy is not on the cell's
// relation, and a cell's own state's is the relation's to rounding.
template <class Recon, class Sides, class Qh>
KOKKOS_INLINE_FUNCTION fpdtype stateEnergy(const Recon &r, const Sides &s,
                                           const bool right, const Qh &qh) {
  const auto &f = right ? s.R : s.L;
  fpdtype composition = 0.0;
  for (int n = 0; n < ns - 1; n++) {
    fpdtype rhoYL, rhoYR;
    r.species(s, n, rhoYL, rhoYR);
    composition += qh(slotZeta + n) * (right ? rhoYR : rhoYL);
  }
  const fpdtype ke =
      0.5 * (f.rhou * f.rhou + f.rhov * f.rhov + f.rhow * f.rhow) / f.rho;
  return qh(slotTheta) * f.p + qh(slotEps) * f.rho + composition +
         qh(slotKappa) + ke;
}

// the internal energy per unit mass flux the two-point central flux
// carries across a face between the conserved states L and R at pressures
// pL and pR, in the relation of qh: eps, the composition part at the face
// mass fraction, the mean of the two sides', and the harmonic mean of the
// two sides' (theta p + kappa) / rho -- at one pressure theta p + kappa
// over the mean density, so a contact at rest moves no energy the relation
// cannot place
template <class QL, class QR, class Qh>
KOKKOS_INLINE_FUNCTION fpdtype faceEnergy(const fpdtype pL, const QL &L,
                                          const fpdtype pR, const QR &R,
                                          const Qh &qh) {
  const fpdtype rhoinvL = 1.0 / L(0), rhoinvR = 1.0 / R(0);
  fpdtype composition = 0.0;
  for (int n = 0; n < ns - 1; n++)
    composition += qh(slotZeta + n) * 0.5 *
                   (L(5 + n) * rhoinvL + R(5 + n) * rhoinvR);
  const fpdtype a = thermal(pL, rhoinvL, qh), b = thermal(pR, rhoinvR, qh);
  return qh(slotEps) + composition + 2.0 * (a * b) / (a + b);
}

} // namespace doubleFlux

#endif
