<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.mixture.transport.kernels.dilute"/>

<%pg:macro name="transport" params="mu, kappa, rho, T, lnT, X, Y, mom">
  ${pg.expand("dilute", "mu, kappa, T, lnT, X")}
</%pg:macro>
