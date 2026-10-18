<%namespace name="pg" module="peregrine.codegen.makoutil"/>

<%pg:macro name="sensors" params="shock, contact"
           state="in at LL, L, R, RR">
  const fpdtype pgain = ${values.get("p-gain", 1.0)}, gammagain = ${values.get("gamma-gain", 1.0)};
  const auto sensor = [](const fpdtype fm, const fpdtype f, const fpdtype fp) { return fabs(fp - 2.0 * f + fm) / fabs(fp + 2.0 * f + fm); };
  const fpdtype pressure = pgain * fmax(sensor(stateLL.p, stateL.p, stateR.p), sensor(stateL.p, stateR.p, stateRR.p));
  ## rho c^2 / p is the effective gamma, whose jump across a contact is what a conservative energy flux cannot carry without a pressure error
  const fpdtype gL = stateL.rho * stateL.c * stateL.c / stateL.p, gR = stateR.rho * stateR.c * stateR.c / stateR.p;
  const fpdtype gamma = gammagain * fabs(gR - gL) / fabs(gR + gL);
  shock = fmin(pressure, 1.0);
  contact = fmin(fmax(pressure, gamma), 1.0);
</%pg:macro>
