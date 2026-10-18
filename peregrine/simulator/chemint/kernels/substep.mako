<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%include file="peregrine.simulator.chemint.kernels.cell"/>

## src is the mean rate over the step, so the integrator's explicit step lands the state the substeps reached
<%pg:macro name="integrate" params="taken, rho, rhoinv, step, Y, T, src, remaining">
  const fpdtype eps = ${finfo.eps}, tiny = ${finfo.tiny}, stepinv = 1.0 / step;
  const auto Yof = [&](const int n) { return Y[n]; };
  fpdtype rate[${ns}];
  int sub = 0;
  bool early = false;
  for (; sub < ${limit} && remaining > ${stepfrac}; sub++) {
    RateState s;
    ${pg.expand("ratestate", "s, rho, Yof, T")}
% if exit and eqtol > 0.0:
    // the first substep's fold carries the destruction rates the exit relaxes by
    fpdtype D[${ns}];
    if (sub == 0) {
      ${pg.expand("sources", "rate, D, s")}
    } else {
      ${pg.expand("production", "rate, s")}
    }
% else:
    ${pg.expand("production", "rate, s")}
% endif
    // the limiting species by cross-multiplying, one quotient at the end; a trace species counts, its substeps keep the fast radicals stable
    fpdtype abest = 0.0, bbest = 0.0;
    for (int n = 0; n < ${ns}; n++) {
      const fpdtype sign = copysign(1.0, rate[n]);
      const fpdtype headroom = 0.5 * ((1.0 - sign) * Y[n] + (1.0 + sign) * (1.0 - Y[n]));
      const fpdtype a = rho * fmax(headroom, eps), b = fmax(fabs(rate[n]), tiny);
      if (n == 0 || a * bbest < abest * b) {
        abest = a;
        bbest = b;
      }
    }
    const fpdtype share = fmin(remaining, abest / bbest * stepinv);
    // the temperature's rate at constant volume: cv, and the internal energy each species' rate carries
    fpdtype cv = 0.0, dTdt = 0.0;
    for (int n = 0; n < ${ns}; n++) {
      fpdtype cpR, e;
      ${pg.expand("cpRspecies", "cpR, n, s.logT")}
      ${pg.expand("especies", "e, n, T, s.logT, s.Tinv")}
      cv += (cpR - 1.0) * Rsp(n) * Y[n];
      dTdt -= e * rate[n];
    }
    const fpdtype cvrhoinv = 1.0 / (cv * rho);
    dTdt *= cvrhoinv;
% if exit and eqtol > 0.0:
    // the equilibrium exit, judged at the first substep only: nothing moves past tolerance over what remains, so the rest is one move with each species relaxed at its destruction frequency, a slow one along its rate, a fast one onto its quasi-steady value
    const fpdtype tleft = remaining * step, tleftinv = 1.0 / tleft;
    bool settled = sub == 0 && tleft > 0.0 && fabs(dTdt) * tleft < 0.1;
    for (int n = 0; n < ${ns} && settled; n++)
      settled = fabs(rate[n] * rhoinv) * tleft < ${eqtol};
    if (settled) {
      fpdtype dT = 0.0;
      for (int n = 0; n < ${ns}; n++) {
        const fpdtype rhoden = rho * fmax(Y[n], cfloor * MW(n) * rhoinv);
        const fpdtype move = fmax((rate[n] * rhoinv) * tleft * rhoden / (rhoden + D[n] * tleft), -Y[n]);
        const fpdtype r = move * rho * tleftinv;
        fpdtype e;
        ${pg.expand("especies", "e, n, T, s.logT, s.Tinv")}
        dT -= e * r;
        Y[n] += move;
        src[n] += r * remaining;
      }
      T += dT * cvrhoinv * tleft;
      remaining = 0.0;
      early = true;
      break;
    }
% endif
    const fpdtype tsub = share * step;
    for (int n = 0; n < ${ns}; n++) {
      Y[n] += (rate[n] * rhoinv) * tsub;
      src[n] += rate[n] * share;
    }
    T += dTdt * tsub;
    remaining -= share;
    if (share <= ${stepfrac}) {
      remaining = 0.0;
      early = true;
      break;
    }
  }
  taken = early ? sub + 1 : sub;
</%pg:macro>

<%pg:macro name="startstate" params="Y, T, src, Q, rhoinv, state">
  ${pg.expand("massfrac", "Yc, Q, rhoinv", scoped=False)}
  for (int n = 0; n < ${ns}; n++) {
    Y[n] = Yc(n);
    src[n] = 0.0;
  }
  T = state.T;
</%pg:macro>
