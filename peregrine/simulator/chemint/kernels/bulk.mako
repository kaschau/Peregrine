<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.chemint.kernels.substep"/>
<%pg:kernel name="bulk" over="cells"
            Q="in [${str(ne)}]"
            state="in"
            dQ="out [${str(ne)}]"
            stiffcells="inout [${str(bins)}] at list"
            substeps="inout"
            dt="in">
  const fpdtype rho = Q[0], rhoinv = 1.0 / rho, step = dt;
  fpdtype Y[${ns}], src[${ns}], T, remaining = 1.0;
  ${pg.expand("startstate", "Y, T, src, Q, rhoinv, state")}
  int taken;
  ${pg.expand("integrate", "taken, rho, rhoinv, step, Y, T, src, remaining")}
% if limit < cap:
  if (remaining > ${stepfrac}) {
% if bins == 1:
    const int b = 0;
% else:
    // binned by the cell's count from the last right-hand side, a cell just turned stiff to the top bin, the ends taking what falls past them
    const int expected = substeps > ${limit} ? substeps : ${cap};
    const int spaced = static_cast<int>(log(fpdtype(expected) * ${limitinv}) * ${binscale});
    const int b = spaced < 0 ? 0 : (spaced >= ${bins} ? ${bins} - 1 : spaced);
% endif
    // counts up the header, [0] every listed cell and [1 + b] bin b's; a bin's segment holds more slots than the interior has cells, so it never fills
    const int slot = pg::atomic_add(stiffcells[1 + b], 1);
    pg::atomic_add(stiffcells[0], 1);
    stiffcells[stiffcells.start(b) + slot] = stiffcells.code();
    return;
  }
% endif
  substeps = taken;
  ${pg.expand("beginsource", "dQ, src")}
</%pg:kernel>
