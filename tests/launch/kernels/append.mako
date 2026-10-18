<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:kernel name="append" over="cells" codes="inout [${str(bins)}] at list" mark="out">
  const int code = codes.code();
  const int b = code % ${bins};
  const int slot = pg::atomic_add(codes[1 + b], 1);
  pg::atomic_add(codes[0], 1);
  codes[codes.start(b) + slot] = code;
  mark = 0;
</%pg:kernel>
