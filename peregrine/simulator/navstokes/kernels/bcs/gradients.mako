<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:macro name="neumann" params="gradshalo, grads">
  for (int l = 0; l < ${ngrad}; l++)
    for (int d = 0; d < 3; d++)
      gradshalo[l][d] = grads[l][d];
</%pg:macro>
