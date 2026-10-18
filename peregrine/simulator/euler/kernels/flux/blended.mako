<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.euler.kernels.flux.recon.${recon}"/>
<%include file="${fluxtpls}.function.${primary}"/>
<%include file="peregrine.simulator.euler.kernels.flux.switch.${switch}"/>
<%include file="${fluxtpls}.function.${secondary}"/>
<%pg:kernel name="flux" over="faces"
            S="in [3]"
            F="out [${str(nflux)}]">
  ${pg.expand("sides", "L, R, YL, YR", scoped=False)}
  fpdtype shock, contact;
  ${pg.expand("sensors", "shock, contact")}
  const fpdtype w = fmax(shock, contact);
  ${pg.expand(primary, "L, R, YL, YR, S, F")}
  fpdtype Fs[${nflux}];
  ${pg.expand(secondary, "L, R, YL, YR, S, Fs")}
  for (int l = 0; l < ${nflux}; l++)
    F[l] = (1.0 - w) * F[l] + w * Fs[l];
</%pg:kernel>
