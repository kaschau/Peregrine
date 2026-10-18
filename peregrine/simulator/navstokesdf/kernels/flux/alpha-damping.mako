<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:alias name="alphadamping" file="peregrine.simulator.navstokes.kernels.flux.alpha-damping" under="plainalphadamping"/>
## f4 is the energy row of the plain flux, left in scope by the unscoped expansion, and is the same in either plane
<%pg:kernel name="flux" over="faces"
            S="in [3]"
            F="inout [${str(nflux)}]">
  ${pg.expand("plainalphadamping", "S, F", scoped=False)}
  F[${ne}] += f4;
</%pg:kernel>
