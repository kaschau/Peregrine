<%namespace name="pg" module="peregrine.codegen.makoutil"/>
## a row follows the privars order, which the host writes as the csv header
<%pg:kernel name="trace" over="listed"
            Q="in [${str(ne)}]"
            state="in"
            tracelist="in at list"
            traced="out at list">
  const fpdtype rhoinv = 1.0 / Q[0];
  const int row = tracelist.slot() * ${ne};
  traced[row] = state.p;
  for (int d = 0; d < 3; d++)
    traced[row + 1 + d] = Q[1 + d] * rhoinv;
  traced[row + 4] = state.T;
  for (int n = 0; n < ${ns} - 1; n++)
    traced[row + 5 + n] = Q[5 + n] * rhoinv;
</%pg:kernel>
