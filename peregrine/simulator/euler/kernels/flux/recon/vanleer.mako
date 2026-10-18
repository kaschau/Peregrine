<%namespace name="pg" module="peregrine.codegen.makoutil"/>

## the limited slope b phi(a / b) of the upwind difference a against the one across the face b, b positive, without forming the ratio
<%pg:macro name="limiter" params="bphi, a, b">
  bphi = (a + fabs(a)) * b / (b + fabs(a));
</%pg:macro>
