<%namespace name="pg" module="peregrine.codegen.makoutil"/>

<%def name="physgrad()">
    for (int e = 0; e < 3; e++)
      grads[l][e] = d[0] * Jinv[0][e] + d[1] * Jinv[1][e] + d[2] * Jinv[2][e];
</%def>
## the conserved row of the quantity sits one past its gradient row; the difference of two quotients is taken as one division
<%def name="fromconserved()">
    const int m = l + 1;
    const fpdtype d[3] = {
        0.5 * (QpI[m] * rIm - QmI[m] * rIp) * invI,
        0.5 * (QpJ[m] * rJm - QmJ[m] * rJp) * invJ,
        0.5 * (QpK[m] * rKm - QmK[m] * rKp) * invK};
    ${physgrad()}
</%def>
<%pg:kernel name="dqdxyz" over="cells"
            Q="in [${str(ne)}] at +I, -I, +J, -J, +K, -K"
            state="in at +I, -I, +J, -J, +K, -K"
            Jinv="in [3, 3]"
            grads="out [${str(ngrad)}, 3]">
  const fpdtype rIp = QpI[0], rIm = QmI[0], invI = 1.0 / (rIp * rIm);
  const fpdtype rJp = QpJ[0], rJm = QmJ[0], invJ = 1.0 / (rJp * rJm);
  const fpdtype rKp = QpK[0], rKm = QmK[0], invK = 1.0 / (rKp * rKm);
  // the velocity, rows 0 .. 2
  for (int l = 0; l < 3; l++) {
    ${fromconserved()}
  }
  // the temperature, row 3, off the state
  {
    const int l = 3;
    const fpdtype d[3] = {
        0.5 * (statepI.T - statemI.T),
        0.5 * (statepJ.T - statemJ.T),
        0.5 * (statepK.T - statemK.T)};
    ${physgrad()}
  }
  // the carried mass fractions, rows 4 on
  for (int l = 4; l < ${ngrad}; l++) {
    ${fromconserved()}
  }
</%pg:kernel>
