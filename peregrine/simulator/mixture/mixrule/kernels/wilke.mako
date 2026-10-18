<%namespace name="pg" module="peregrine.codegen.makoutil"/>
## phi_nm = (1 + s_n / s_m)^2 wilkepair(n, m), with s_n = sqrt(mu_n) / MW_n^(1/4) and wilkepair the baked 1 / sqrt(8 (1 + MW_n / MW_m))
<%pg:macro name="viscmix" params="mu, X, sqrtmu">
  fpdtype sinv[${ns}];
  for (int n = 0; n < ${ns}; n++)
    sinv[n] = 1.0 / (sqrtmu[n] * MWqinv(n));
  mu = 0.0;
  for (int n = 0; n < ${ns}; n++) {
    const fpdtype sn = sqrtmu[n] * MWqinv(n);
    fpdtype phi = 0.0;
    for (int m = 0; m < ${ns}; m++) {
      const fpdtype r = 1.0 + sn * sinv[m];
      phi += r * r * wilkepair(n, m) * X(m);
    }
    mu += sqrtmu[n] * sqrtmu[n] * X(n) / phi;
  }
</%pg:macro>
