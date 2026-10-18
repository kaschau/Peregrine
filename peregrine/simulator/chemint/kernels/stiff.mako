<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.chemint.kernels.substep"/>
## restarts from the start rather than resuming the bulk pass, so a stiff cell's result is the one-pass form's to the bit
<%pg:kernel name="stiff" over="listed"
            Q="in [${str(ne)}]"
            state="in"
            dQ="out [${str(ne)}]"
            stiffcells="in [${str(bins)}] at list"
            substeps="inout"
            dt="in">
  const fpdtype rho = Q[0], rhoinv = 1.0 / rho, step = dt;
  fpdtype Y[${ns}], src[${ns}], T, remaining = 1.0;
  ${pg.expand("startstate", "Y, T, src, Q, rhoinv, state")}
  int taken;
  ${pg.expand("integrate", "taken, rho, rhoinv, step, Y, T, src, remaining")}
  substeps = taken;
  ${pg.expand("beginsource", "dQ, src")}
</%pg:kernel>
