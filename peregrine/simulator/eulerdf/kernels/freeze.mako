<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="${eos}"/>
<%pg:kernel name="freeze" over="cells"
            Q="inout [${str(ne)}]"
            state="inout">
  const fpdtype rho = Q[0], rhoinv = 1.0 / rho;
  const fpdtype p = state.p, T = state.T, e = state.e;
  const fpdtype rhoe = rho * e;
  // the energy reset to the eos's: a step in the frozen plane leaves it off the eos
  Q[4] = rhoe + 0.5 * ${pg.dot("Q[{i}]", i=(1, 4))} * rhoinv;
  // the carried mass fractions, as the eos reads them
  const auto Y = [&](const int n) { return Q[5 + n] * rhoinv; };
  MixMom mom;
  ${pg.expand("mixmom", "mom, Y")}
  TangentPlane plane;
  fpdtype epsY;
  ${pg.expand("planeslopes", "plane.theta, epsY, mom, Y, rho, rhoinv, T, e")}
  fpdtype rho_p, rho_T, rho_Y[${ns}];
  ${pg.expand("drho", "rho_p, rho_T, rho_Y, mom, Y, p, T, rho")}
  fpdtype h[${ns}];
  ${pg.expand("hspecies", "h, mom, Y, state")}
  const fpdtype cpoverrhoT = state.cp / rho_T;
  fpdtype zetaY = 0.0;
  // the species slope at fixed rho and p, the last species giving way: the partial enthalpy difference less cp times the T change that holds rho
% for n in range(ns - 1):
  plane.zeta${n} = h[${n}] - h[${ns} - 1] - cpoverrhoT * rho_Y[${n}];
  zetaY += plane.zeta${n} * Y(${n});
% endfor
  plane.eps = epsY - zetaY;
  plane.kappa = rhoe - epsY * rho - plane.theta * p;
  state.store(plane);
</%pg:kernel>
