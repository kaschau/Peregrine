<%namespace name="pg" module="peregrine.codegen.makoutil"/>

## Reads the velocity gradients, so it is the viscous flow's: under Euler, which has none, it fails as it is rendered
<%pg:macro name="sensors" params="shock, contact"
           Q="in [${str(ne)}] at L, R"
           grads="in [${str(ngrad)}, 3] at L, R"
           dijk="in [3] at L, R">
  const fpdtype nu = ${values.get("nu", 0.1)}, least = ${values.get("floor", 0.0)};
  ## 0 / 0 in a quiescent flow guards to none
  const fpdtype tiny = ${finfo.tiny};
  ## g is the velocity gradient, rows u, v, w by columns x, y, z
  const auto sensor = [&](const auto &g, const auto &Qc, const auto &d) {
    const fpdtype div = g[0][0] + g[1][1] + g[2][2];
    const fpdtype div2 = div * div;
    const fpdtype wx = g[2][1] - g[1][2], wy = g[0][2] - g[2][0], wz = g[1][0] - g[0][1];
    const fpdtype omega2 = wx * wx + wy * wy + wz * wz;
    const fpdtype D = fmin(4.0 / 3.0 * div2 / (div2 + omega2 + tiny), 1.0);
    const fpdtype rhoinv = 1.0 / Qc[0];
    const fpdtype u2 = (Qc[1] * Qc[1] + Qc[2] * Qc[2] + Qc[3] * Qc[3]) * rhoinv * rhoinv;
    const fpdtype l = 0.5 * fmin(fmin(d[0], d[1]), d[2]);
    const fpdtype cutoff2 = nu * nu * u2 / (l * l);
    const fpdtype theta = div2 / (div2 + cutoff2 + tiny);
    return fmax(D - least, 0.0) * theta + least;
  };
  shock = fmax(sensor(gradsL, QL, dijkL), sensor(gradsR, QR, dijkR));
  contact = shock;
</%pg:macro>
