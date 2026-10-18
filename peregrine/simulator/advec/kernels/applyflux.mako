<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:kernel name="applyflux" over="cells"
            detJinv="in"
            iF="in [${str(nflux)}] at low, high"
            jF="in [${str(nflux)}] at low, high"
            kF="in [${str(nflux)}] at low, high"
            dQ="${'inout' if append else 'out'} [${str(ne)}]">
<% op = "+=" if append else "=" %>
  for (int l = 0; l < ${ne}; l++)
    dQ[l] ${op} (iFlow[l] + jFlow[l] + kFlow[l] - iFhigh[l] - jFhigh[l] - kFhigh[l]) * detJinv;
</%pg:kernel>
