<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%! from peregrine.backend.base.mathfns import EXPBINS, EXPBITS, EXPK, EXPTAB, LN2, LN2BINS_HI, LN2BINS_LO, LOGBINS, LOGBITS, LOGTAB, RCPTAB %>
${pg.constants(logtab=LOGTAB, rcptab=RCPTAB, exptab=EXPTAB)}
## tlog: log x by a table over the mantissa and a series, within 3e-14 absolute, two ulps past |log x| = 100; in single the library's
## no relative accuracy near x = 1, no sign, NaN or inf: zero and a denormal come out near -700, where a floor takes over, inf and NaN finite near 710
<%pg:macro name="tlog" params="y, x">
% if finfo.bits < 64:
  y = log(x);
% else:
  const double v = x;
  long long bits;
  memcpy(&bits, &v, sizeof bits);
  const int expo = (int)((bits >> 52) & 0x7ff) - 1022;
  const int bin = (int)((bits >> ${52 - LOGBITS}) & ${LOGBINS - 1});
  // the mantissa as a double in [0.5, 1)
  bits = (bits & 0x000fffffffffffffLL) | 0x3fe0000000000000LL;
  double mant;
  memcpy(&mant, &bits, sizeof mant);
  const double u = mant * rcptab(bin) - 1.0;
  double series = 0.2;
  series = series * u - 0.25;
  series = series * u + 1.0 / 3.0;
  series = series * u - 0.5;
  series = series * u + 1.0;
  y = expo * ${LN2.hex()} + logtab(bin) + u * series;
% endif
</%pg:macro>

## texp: exp x by a table over the fractional powers of two and a series, within two ulps; in single the library's
## below x = -708.4 a tiny positive number near 2e-308, never zero; above 709 not finite, so a site clamps its argument; inf and NaN come out finite
<%pg:macro name="texp" params="y, x">
% if finfo.bits < 64:
  y = exp(x);
% else:
  const double v = x;
  // 1.5 2^52 added leaves the nearest integer in the low word
  const double t = v * ${EXPK.hex()} + 6755399441055744.0;
  long long bits;
  memcpy(&bits, &t, sizeof bits);
  const int kexp = (int)bits, nexp = kexp >> ${EXPBITS};
  const double kd = t - 6755399441055744.0;
  const double r = (v - kd * ${LN2BINS_HI.hex()}) - kd * ${LN2BINS_LO.hex()};
  double series = 1.0 / 120.0;
  series = series * r + 1.0 / 24.0;
  series = series * r + 1.0 / 6.0;
  series = series * r + 0.5;
  series = series * r + 1.0;
  series = series * r + 1.0;
  const double m = exptab(kexp & ${EXPBINS - 1}) * series;
  memcpy(&bits, &m, sizeof bits);
  bits += (long long)(nexp > -1022 ? nexp : -1022) << 52;
  memcpy(&y, &bits, sizeof bits);
% endif
</%pg:macro>
