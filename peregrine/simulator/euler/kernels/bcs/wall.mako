<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:macro name="wallflux" params="F, p, S">
  F[0] = 0.0;
  F[1] = p * S[0];
  F[2] = p * S[1];
  F[3] = p * S[2];
  for (int l = 4; l < ${nflux}; l++)
    F[l] = 0.0;
</%pg:macro>

## the acoustic estimate p + rho c u_n, a Riemann solver's star pressure to first order, u_n the cell's speed toward the wall
<%pg:macro name="wallpressure" params="pw, Q, state, outward, S">
  const fpdtype area = sqrt(${pg.dot("S[{i}]", i=3)});
  const fpdtype un = outward * ${pg.dot("Q[{i}]", "S[{j}]", i=(1, 4), j=3)} / (Q[0] * area);
  pw = fmax(state.p + Q[0] * state.c * un, 0.0);
</%pg:macro>
