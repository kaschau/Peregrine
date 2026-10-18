<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%! import math %>
<%include file="peregrine.simulator.mixture.eos.kernels.species"/>
<%include file="peregrine.backend.base.kernels.mathfns"/>
## folded in log space, floored and clamped, so an absent species and a hot cell stay finite in any precision

<%def name="progress(k, r)">
    // reaction ${k}, ${r.name}${", reversible" if r.reversible else ""}
    fpdtype lk = rxnlogA(${k});
% if r.b != 0.0:
    lk += rxnb(${k}) * s.logT;
% endif
% if r.EaR != 0.0:
    lk -= rxnEaR(${k}) * s.Tinv;
% endif
% if r.thirdbody and not r.falloff:
    lk += s.logtb[${r.tb}];
% endif
% if r.falloff:
    // ln Pr, and ln(Pr / (1 + Pr)) in softplus form, exact and finite for any ln Pr
    fpdtype logpr = s.logtb[${r.tb}] + rxnlogprA(${k});
% if r.prb != 0.0:
    logpr += rxnprb(${k}) * s.logT;
% endif
% if r.prEaR != 0.0:
    logpr -= rxnprEaR(${k}) * s.Tinv;
% endif
    fpdtype prexp;
    ${pg.expand("texp", "prexp, -fabs(logpr)")}
    lk += fmin(logpr, 0.0) - log1p(prexp);
% endif
% if r.troe:
${troeblend(r)}
% endif
    // the forward rate of progress, ln
    fpdtype lrp = lk;
% for j, (sp, _) in enumerate(r.fwd):
    lrp += rxnfwdorder(${k}, ${j}) * s.logc[${sp}];
% endfor
% if r.reversible:
    // the affinity ln(forward / reverse)
    fpdtype diff = ${f"rxnnusum({k}) * s.logprefRuT" if r.nusum != 0.0 else "0.0"};
% for j, (sp, _, _) in enumerate(r.net):
    diff -= rxnnetnu(${k}, ${j}) * (s.gRT[${sp}] + s.logc[${sp}]);
% endfor
% endif
</%def>

## Troe's broadening added to ln k
<%def name="troeblend(r)">
% if not r.fcent:
    // a mechanism whose centre terms cancel is floored as Cantera floors it
    const fpdtype lfc = ${math.log(float(finfo.tiny)) * 0.4342944819032518};
% else:
    // Troe's centre by its polynomial fit in ln T over the case's temperatures
    const fpdtype tf = (s.logT - troeuc) * troeuhinv;
    fpdtype lfc = rxnlfcpoly(${r.troeidx}, ${len(r.lfcpoly) - 1});
% for m in reversed(range(len(r.lfcpoly) - 1)):
    lfc = lfc * tf + rxnlfcpoly(${r.troeidx}, ${m});
% endfor
% endif
    const fpdtype C = -0.4 - 0.67 * lfc, N = 0.75 - 1.27 * lfc;
    // the blend 1 / (1 + (A / B)^2) as B^2 / (B^2 + A^2): one division
    const fpdtype A = logpr * log10e + C, B = N - 0.14 * A, B2 = B * B;
    lk += lfc * B2 / (B2 + A * A) * ln10;
</%def>

## the product form: one exponential K = k over the reactants' e^(g / Ru T), each one-way rate K times its side's c e^(g / Ru T), the reverse's times (pref / Ru T)^-nusum
<%def name="kform(k, r)">
    // reaction ${k}, ${r.name}${", reversible" if r.reversible else ""}, product form
    fpdtype lk = rxnlogA(${k});
% if r.b != 0.0:
    lk += rxnb(${k}) * s.logT;
% endif
% if r.EaR != 0.0:
    lk -= rxnEaR(${k}) * s.Tinv;
% endif
% if r.falloff:
    fpdtype logpr = s.logtb[${r.tb}] + rxnlogprA(${k});
% if r.prb != 0.0:
    logpr += rxnprb(${k}) * s.logT;
% endif
% if r.prEaR != 0.0:
    logpr -= rxnprEaR(${k}) * s.Tinv;
% endif
    fpdtype pr;
    ${pg.expand("texp", "pr, fmin(logpr, expclamp)")}
    const fpdtype pfac = pr / (1.0 + pr);
% endif
% if r.troe:
${troeblend(r)}
% endif
    fpdtype xk = lk;
% for j, (sp, _) in enumerate(r.react):
    xk -= rxnreactnu(${k}, ${j}) * s.gRT[${sp}];
% endfor
    fpdtype K;
    ${pg.expand("texp", "K, fmin(xk, expclamp)")}
% if r.thirdbody and not r.falloff:
    K *= s.ctb[${r.tb}];
% endif
% if r.falloff:
    K *= pfac;
% endif
    fpdtype fwd = K;
% for sp, nu in r.react:
% for _ in range(int(nu)):
    fwd *= s.cG[${sp}];
% endfor
% endfor
% if r.reversible:
    fpdtype rev = K;
% for _ in range(int(abs(r.nusum))):
    rev *= ${"s.prefRuTinv" if r.nusum > 0 else "s.prefRuT"};
% endfor
% for sp, nu in r.prod:
% for _ in range(int(nu)):
    rev *= s.cG[${sp}];
% endfor
% endfor
% else:
    const fpdtype rev = 0.0;
% endif
    const fpdtype rp = fwd - rev;
</%def>

<%def name="prelude()">
% if rxns.troes:
  const fpdtype log10e = 0.4342944819032518, ln10 = 2.302585092994046;
% endif
  // the largest ln rate of progress whose production rates stay finite
  const fpdtype expclamp = ${math.log(float(finfo.max)) - math.log(2.0 * rxns.maxgain)};
</%def>

## the log form's logs for the reactions in it, single precision's all; the product form's c e^(g / Ru T) in double for the rest
<%pg:macro name="ratestate" params="s, rho, Y, T">
<%
  logform = finfo.bits < 64 or rxns.logform
  product = finfo.bits == 64 and rxns.product
%>
  s.T = T;
  ${pg.expand("tlog", "s.logT, T")}
  s.Tinv = 1.0 / T;
% if logform:
  s.logprefRuT = logprefRu - s.logT;
% endif
% if product:
  s.prefRuT = prefRu * s.Tinv;
  s.prefRuTinv = Rupref * T;
% endif
  s.ctot = 0.0;
  for (int n = 0; n < ${ns}; n++) {
    const fpdtype c = rho * Y(n) * MWinv(n);
    s.ctot += c;
    ${pg.expand("gRTspecies", "s.gRT[n], n, s.logT, s.Tinv")}
% if logform:
    // log of nothing is below the floor: fmax takes the number
    ${pg.expand("tlog", "s.logc[n], c")}
    s.logc[n] = fmax(logcfloor, s.logc[n]);
% endif
% if product:
    // a substep can leave a mass fraction a rounding below zero: the floor keeps a one-way rate's sign
    fpdtype E;
    ${pg.expand("texp", "E, s.gRT[n]")}
    s.cG[n] = fmax(c, 0.0) * E;
% endif
  }
% for j, tb in enumerate(rxns.thirdbodies):
  {
    fpdtype ctbc = tbdefeff(${j}) * s.ctot;
% for i, (sp, _) in enumerate(tb.eff):
    ctbc += tbeffdev(${j}, ${i}) * (rho * Y(${sp}) * MWinv(${sp}));
% endfor
% if product and j in rxns.plainbodies:
    s.ctb[${j}] = ctbc;
% endif
% if finfo.bits < 64 or j in rxns.logbodies:
    ${pg.expand("tlog", f"s.logtb[{j}], fmax({finfo.tiny}, ctbc)")}
% endif
  }
% endfor
</%pg:macro>

## one fold for both: omega the net production rate and, asked for, D the destruction rate, kg / m^3 / s
<%def name="fold(omega, D)">
${prelude()}
  for (int n = 0; n < ${ns}; n++)
    ${omega}[n] = 0.0;
% if D:
  for (int n = 0; n < ${ns}; n++)
    ${D}[n] = 0.0;
% endif
% for k, r in enumerate(rxns.reactions):
  {
% if finfo.bits == 64 and r.product:
${kform(k, r)}
% else:
${progress(k, r)}
% if r.reversible and finfo.bits < 64:
    // single: the surviving direction times expm1 of the affinity, a subtraction would lose a near-equilibrium net rate to rounding
    const fpdtype rp = copysign(exp(fmin(lrp - fmin(diff, 0.0), expclamp)), diff) * (-expm1(-fabs(diff)));
% if D:
    const fpdtype fwd = exp(fmin(lrp, expclamp)), rev = exp(fmin(lrp - diff, expclamp));
% endif
% elif r.reversible:
    // double: forward less reverse, the one-way rates' rounding is below the net rate's share of a cell's largest
    fpdtype fwd, rev;
    ${pg.expand("texp", "fwd, fmin(lrp, expclamp)")}
    ${pg.expand("texp", "rev, fmin(lrp - diff, expclamp)")}
    const fpdtype rp = fwd - rev;
% else:
    fpdtype fwd;
    ${pg.expand("texp", "fwd, fmin(lrp, expclamp)")}
    const fpdtype rev = 0.0;
    const fpdtype rp = fwd;
% endif
% endif
% for j, (sp, _, nuMW) in enumerate(r.net):
    ${omega}[${sp}] += rxnnetnuMW(${k}, ${j}) * rp;
% if D:
    ${D}[${sp}] += ${f"-rxnnetnuMW({k}, {j})" if nuMW < 0 else "0.0"} * fwd + ${f"rxnnetnuMW({k}, {j})" if nuMW > 0 else "0.0"} * rev;
% endif
% endfor
  }
% endfor
</%def>

## omega in kg / m^3 / s
<%pg:macro name="production" params="omega, s">
${fold("omega", None)}
</%pg:macro>

## omega and D in kg / m^3 / s, one fold
<%pg:macro name="sources" params="omega, D, s">
${fold("omega", "D")}
</%pg:macro>
