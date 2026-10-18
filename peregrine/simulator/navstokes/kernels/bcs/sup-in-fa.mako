<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%pg:alias name="preadvflux" file="peregrine.simulator.euler.kernels.bcs.sup-in-fa"/>
<%include file="peregrine.simulator.navstokes.kernels.bcs.gradients"/>
<%pg:kernel name="postgrads" over="halocells"
            grads="in [${str(ngrad)}, 3] at mirror"
            gradshalo="out [${str(ngrad)}, 3] at halo">
  ${pg.expand("neumann", "gradshalo, grads")}
</%pg:kernel>
