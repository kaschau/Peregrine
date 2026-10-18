<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:kernel name="preadvflux" over="halocells"
            vals="in [${str(ne)}] at plane"
            Qhalo="out [${str(ne)}] at halo">
  for (int l = 0; l < ${ne}; l++)
    Qhalo[l] = vals[l];
</%pg:kernel>
