<%namespace name="pg" module="peregrine.codegen.makoutil"/>
## runs after the transport kernel in its slot, so trstate[0] is the viscosity that kernel has just made
<%pg:kernel name="viscsponge" over="cells" params="nx, ny, nz, start, length, mult"
            xcc="in [3]"
            trstate="inout [${str(2 + ns)}]">
  const fpdtype along = (${pg.dot("xcc[{i}]", "n{c}", i=3, c="xyz")} - start) / length;
  trstate[0] *= 1.0 + fmin(fmax(0.0, along * (mult - 1.0)), mult - 1.0);
</%pg:kernel>
