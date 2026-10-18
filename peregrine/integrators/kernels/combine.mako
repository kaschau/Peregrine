<%namespace name="pg" module="peregrine.codegen.makoutil"/>
<%
reads = [f"w{n} * {'dt * ' if step and n == terms - 1 else ''}X{n}[l]" for n in range(terms)]
rhs = (["wout * out[l]"] if own else []) + reads
decls = {"out": f"{'inout' if own else 'out'} [{ne}]", **{f"X{n}": f"in [{ne}]" for n in range(terms)}}
if step:
    decls["dt"] = "in"
params = ", ".join((["wout"] if own else []) + [f"w{n}" for n in range(terms)])
%>
<%pg:kernel name="combine" over="cells" decls="${decls}" params="${params}">
  for (int l = 0; l < ${ne}; l++)
    out[l] = ${" + ".join(rhs)};
</%pg:kernel>
