<%namespace name="pg" module="peregrine.codegen.makoutil"/>

## the limited slope b phi(a / b) of the upwind difference a against the one across the face b, b positive
<%pg:macro name="limiter" params="bphi, a, b">
  bphi = fmax(0.0, fmin(fmin(2.0 * a, (2.0 * b + a) * ${1.0 / 3.0}), 2.0 * b));
</%pg:macro>
