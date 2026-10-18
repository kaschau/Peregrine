<%namespace name="pg" module="peregrine.codegen.makoutil"/>
## the list's header: [0] every listed cell, [1 + b] bin b's count
<%pg:kernel name="reset" over="cells"
            stiffcells="inout [${str(bins)}] at list">
  for (int b = 0; b <= ${bins}; b++)
    stiffcells[b] = 0;
</%pg:kernel>
