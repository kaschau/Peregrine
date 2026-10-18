<%namespace name="pg" module="peregrine.codegen.makoutil"/>
## finite is read off the exponent bits: a build that assumes finite arithmetic folds isfinite() to true, and a device build may
<%pg:kernel name="allfinite" over="cells" reduce="finite min [1]"
            Q="in [${str(ncomps)}]">
% if fpdtype == "float":
  const uint32_t exponent = 0x7f800000u;
% else:
  const uint64_t exponent = 0x7ff0000000000000ull;
% endif
  for (int l = 0; l < ${ncomps}; l++) {
    const fpdtype x = Q[l];
% if fpdtype == "float":
    const bool ok = (pg::bits(x) & exponent) != exponent;
% else:
    const bool ok = (pg::bits(x) & exponent) != exponent;
% endif
    finite[0] = fmin(finite[0], ok ? 1.0 : 0.0);
  }
</%pg:kernel>
