<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.euler.kernels.flux.recon.${recon}"/>
## the cells read are the reconstruction's to declare
<%pg:kernel name="sides" over="faces"
            S="in [3]"
            out="out [${str(2 * (8 + ns - 1))}]">
  ${pg.expand("sides", "L, R, YL, YR", scoped=False)}
% for n, X in enumerate("LR"):
  {
    const int o = ${n} * (8 + ${ns} - 1);
    out[o + 0] = ${X}.p; out[o + 1] = ${X}.T; out[o + 2] = ${X}.rho; out[o + 3] = ${X}.e;
    out[o + 4] = ${X}.c; out[o + 5] = ${X}.u; out[o + 6] = ${X}.v; out[o + 7] = ${X}.w;
    for (int n = 0; n < ${ns} - 1; n++) out[o + 8 + n] = Y${X}(n);
  }
% endfor
</%pg:kernel>
