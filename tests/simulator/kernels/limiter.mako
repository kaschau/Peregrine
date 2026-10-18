<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.euler.kernels.flux.recon.${limiter}"/>
<%pg:kernel name="limiter" over="faces"
            ratio="in [1]"
            out="out [1]">
  // the limited slope at a unit slope across the face is phi of the ratio itself
  const fpdtype r = ratio[0];
  fpdtype phi;
  ${pg.expand("limiter", "phi, r, 1.0")}
  out[0] = phi;
</%pg:kernel>
