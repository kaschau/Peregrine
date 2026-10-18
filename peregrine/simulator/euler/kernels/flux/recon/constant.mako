<%namespace name="pg" module="peregrine.codegen.makoutil"/>

<%pg:macro name="fromcell" params="X, Qc, statec">
  statec.load(X);
  X.u = Qc[1] / Qc[0];
  X.v = Qc[2] / Qc[0];
  X.w = Qc[3] / Qc[0];
</%pg:macro>

<%pg:macro name="sides" params="L, R, YL, YR"
           Q="in [${str(ne)}] at L, R"
           state="in at L, R">
  FaceState L, R;
  ${pg.expand("fromcell", "L, QL, stateL")}
  ${pg.expand("fromcell", "R, QR, stateR")}
  const auto YL = [&](const int n) { return QL[5 + n] / QL[0]; };
  const auto YR = [&](const int n) { return QR[5 + n] / QR[0]; };
</%pg:macro>
