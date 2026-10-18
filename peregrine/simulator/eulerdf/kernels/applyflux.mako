<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:kernel name="applyflux" over="cells"
            detJinv="in"
            iF="in [${str(nflux)}] at low, high"
            jF="in [${str(nflux)}] at low, high"
            kF="in [${str(nflux)}] at low, high"
            dQ="${'inout' if append else 'out'} [${str(ne)}]">
<% op = "+=" if append else "=" %>
% for l in [l for l in range(ne) if l != 4]:
  dQ[${l}] ${op} (iFlow[${l}] + jFlow[${l}] + kFlow[${l}] - iFhigh[${l}] - jFhigh[${l}] - kFhigh[${l}]) * detJinv;
% endfor
  ## the cell's own energy row: F[ne] on its low faces, where it is the right cell, F[4] on its high faces
  dQ[4] ${op} (iFlow[${ne}] + jFlow[${ne}] + kFlow[${ne}] - iFhigh[4] - jFhigh[4] - kFhigh[4]) * detJinv;
</%pg:kernel>
