<%namespace name="pg" module="peregrine.codegen.makoutil"/>

## the bath is the species the case names, or every species; a row's width is the loop's bound as a literal, which the compiler unrolls where a bound it reads leaves the loop rolled with each pair's loads exposed
<%
bath = [int(m) for m in constants["bath"]]
partial = len(bath) < ns
width = len(bath) if partial else ns - 1
%>
## the pair's fit is T^(3/2) / D_nm, so the rule divides by no pair; its row is the species' own, slot k of the walk, so the pair's order never comes up
## each pair's fit is evaluated from both sides: an array of per-species sums is indexed by the inner loop, lands in local memory, and costs more than the repeat
<%pg:macro name="mixavg" params="D, p, T, lnT, MWmix, X, invfactor">
  const fpdtype invT32 = 1.0 / (T * sqrt(T));
% if partial:
  const int bathof[${width}] = {${", ".join(str(m) for m in bath)}};
% endif
  for (int n = 0; n < ${ns}; n++) {
    fpdtype sum1 = 0.0, sum2 = 0.0, rest = 0.0, bathx = 0.0;
    // the species' partners, its row of fits walked in order: every species but itself, stepped over by the index, or the bath's, its own slot weighted out
    for (int k = 0; k < ${width}; k++) {
      // the fraction clipped at zero: a negative trace lets rest cancel while sum1 stays tiny and positive, and D comes out of order 1e36
% if partial:
      const int m = bathof[k];
      const fpdtype Xm = m != n ? fmax(X(m), 0.0) : 0.0;
% else:
      const int m = k + (k >= n);
      const fpdtype Xm = fmax(X(m), 0.0);
% endif
      fpdtype Dinv = 0.0;
      for (int t = pairfit_terms - 1; t >= 0; t--)
        Dinv = Dinv * lnT + pairfit(n * ${width} + k, t);
      Dinv *= invT32 * invfactor(n, m);
      sum1 += Xm * Dinv;
      sum2 += Xm * MW(m) * Dinv;
      // rest is its own sum, not MWmix - MW_n X_n, which cancels to rounding near a pure species
      rest += Xm * MW(m);
      bathx += Xm;
    }
    // the species' own pair: its self-diffusion coefficient's reciprocal at p
    fpdtype Dinvself = 0.0;
    for (int t = selffit_terms - 1; t >= 0; t--)
      Dinvself = Dinvself * lnT + selffit(n, t);
    Dinvself *= invT32 * invfactor(n, n);
    // one quotient per species, the rule multiplied through by its sums with the operands chosen by a select: a species alone takes its self-diffusion coefficient; alone is X_n within 64 epsilon of one, a few ulps past the rounding a mass fraction comes off the solution with, and a composition clipped to nothing at all gets D = 0
    const fpdtype Xn = fmax(X(n), 0.0);
% if partial:
    // a partial bath leaves partners out, taken to diffuse as the bath does: the first sum scaled from the bath's fraction to the whole, and a species with none of the bath's others present, within 64 epsilon, is alone as well
    const bool alone = 1.0 - Xn <= ${64 * finfo.eps} || bathx <= ${64 * finfo.eps};
    const fpdtype num = alone ? 1.0 : rest * bathx;
    const fpdtype den = alone ? p * Dinvself : p * ((1.0 - Xn) * rest * sum1 + Xn * bathx * sum2);
% else:
    const bool alone = 1.0 - Xn <= ${64 * finfo.eps};
    const fpdtype num = alone ? 1.0 : rest;
    const fpdtype den = alone ? p * Dinvself : p * (rest * sum1 + Xn * sum2);
% endif
    D[n] = (alone || (rest > 0.0 && den > 0.0)) ? num / den : 0.0;
  }
</%pg:macro>

<%pg:macro name="diffcoeffs" params="D, p, T, lnT, rhoinv, cp, kappa, MWmix, X">
  const auto unity = [](const int, const int) { return fpdtype(1.0); };
  ${pg.expand("mixavg", "D, p, T, lnT, MWmix, X, unity")}
</%pg:macro>
