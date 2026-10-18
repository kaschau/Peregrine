<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%
if recon != "constant":
    raise ValueError(f"kepaec4 takes the constant reconstruction, the four cells about the face as they are, not {recon!r}")
%>
<%include file="peregrine.simulator.euler.kernels.flux.recon.constant"/>
<%include file="peregrine.simulator.euler.kernels.flux.function.kepaec"/>
<%pg:macro name="kepaec4" params="L, R, YL, YR, S, F"
           Q="in [${str(ne)}] at LL, RR"
           state="in at LL, RR">
  FaceState LL, RR;
  ${pg.expand("fromcell", "LL, QLL, stateLL")}
  ${pg.expand("fromcell", "RR, QRR, stateRR")}
  const auto YLL = [&](const int n) { return QLL[5 + n] / QLL[0]; };
  const auto YRR = [&](const int n) { return QRR[5 + n] / QRR[0]; };
  fpdtype Fn[${ne}], Fl[${ne}], Fr[${ne}];
  ${pg.expand("kepaec", "L, R, YL, YR, S, Fn")}
  ${pg.expand("kepaec", "LL, R, YLL, YR, S, Fl")}
  ${pg.expand("kepaec", "L, RR, YL, YRR, S, Fr")}
  ## 4/3 Fn - 1/6 (Fl + Fr), written as Fn plus a third of its excess so it is Fn bitwise where the three pairs agree
  for (int l = 0; l < ${ne}; l++)
    F[l] = Fn[l] + (Fn[l] - 0.5 * (Fl[l] + Fr[l])) / 3.0;
</%pg:macro>
