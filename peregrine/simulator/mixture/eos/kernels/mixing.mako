<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<% momstruct = structs["MixMom"] %>
<%pg:macro name="mixmom" params="mom, Y">
% for member in momstruct.fields:
  mom.${member} = 0.0;
% endfor
% for member, terms in momstruct.arrays.items():
  for (int m = 0; m < ${terms}; m++)
    mom.${member}[m] = 0.0;
% endfor
  fpdtype Ylast = 1.0;
  for (int n = 0; n < ${ns} - 1; n++) {
    const fpdtype Yn = Y(n);
    Ylast -= Yn;
% for member in momstruct.fields:
    mom.${member} += Yn * ${moments[member]}(n);
% endfor
% for member, terms in momstruct.arrays.items():
    for (int m = 0; m < ${terms}; m++)
      mom.${member}[m] += Yn * ${moments[member]}(n, m);
% endfor
  }
% for member in momstruct.fields:
  mom.${member} += Ylast * ${moments[member]}(${ns} - 1);
% endfor
% for member, terms in momstruct.arrays.items():
  for (int m = 0; m < ${terms}; m++)
    mom.${member}[m] += Ylast * ${moments[member]}(${ns} - 1, m);
% endfor
</%pg:macro>

## Y(n) runs over every species here, the last included; mixmom's Y(n) is the carried ns - 1
## the mixture's molecular weight, the one factor a mole fraction needs beyond its mass fraction: X_n = Y_n MWmix / MW_n, which the caller spells as a function of the species like Y
<%pg:macro name="molmass" params="MWmix, Y">
  fpdtype denom = 0.0;
  for (int n = 0; n < ${ns}; n++)
    denom += Y(n) * MWinv(n);
  MWmix = 1.0 / denom;
</%pg:macro>
