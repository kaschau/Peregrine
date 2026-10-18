<%namespace name="pg" module="peregrine.codegen.makoutil"/>
## the two views are laid out alike by the exchange: the thread's cell in one is its cell in the other, whatever the alignment of the block across
<% dims = ", ".join(map(str, comps)) %>
<%pg:macro name="rotate" params="v, R, start">
  const fpdtype x = v[start], y = v[start + 1], z = v[start + 2];
  v[start] = R[0][0] * x + R[0][1] * y + R[0][2] * z;
  v[start + 1] = R[1][0] * x + R[1][1] * y + R[1][2] * z;
  v[start + 2] = R[2][0] * x + R[2][1] * y + R[2][2] * z;
</%pg:macro>

<%pg:kernel name="place" over="slab" decls="${dict(rot='in [3, 3] at broadcast') if rotate else {}}"
            dst="${'inout' if rotate else 'out'} [${dims}]" src="in [${dims}]">
% if len(comps) == 1:
  for (int l = 0; l < ${comps[0]}; l++)
    dst[l] = src[l];
% else:
  for (int l = 0; l < ${comps[0]}; l++)
    for (int m = 0; m < ${comps[1]}; m++)
      dst[l][m] = src[l][m];
% endif
% if rotate:
% if len(comps) == 2:
  for (int l = 0; l < ${comps[0]}; l++) {
    ${pg.expand("rotate", "dst[l], rot, 0")}
  }
  // the tensor's component index, down the three rows of every column
  for (int m = 0; m < ${comps[1]}; m++) {
    const fpdtype x = dst[${vecstart}][m], y = dst[${vecstart} + 1][m], z = dst[${vecstart} + 2][m];
    dst[${vecstart}][m] = rot[0][0] * x + rot[0][1] * y + rot[0][2] * z;
    dst[${vecstart} + 1][m] = rot[1][0] * x + rot[1][1] * y + rot[1][2] * z;
    dst[${vecstart} + 2][m] = rot[2][0] * x + rot[2][1] * y + rot[2][2] * z;
  }
% else:
  ${pg.expand("rotate", f"dst, rot, {vecstart}")}
% endif
% endif
</%pg:kernel>
