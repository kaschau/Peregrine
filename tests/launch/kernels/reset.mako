<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:kernel name="reset" over="cells" codes="inout [${str(bins)}] at list">
  for (int b = 0; b <= ${bins}; b++)
    codes[b] = 0;
</%pg:kernel>
