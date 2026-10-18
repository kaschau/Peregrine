<%
    declared = {s.name: s for a in arrays.values() for s in a.structs}
    declared.update(structs)
%>
% for s in declared.values():
struct ${s.name}${f" : {s.base}" if s.base else ""} {
% for field in s.fields:
  fpdtype ${field};
% endfor
% for name, length in s.arrays.items():
  fpdtype ${name}[${length}];
% endfor
};
% endfor
namespace pg {
% for spec in [a for a in arrays.values() if a.structs]:
template <class F> struct ${spec.accessor} {
% for field in spec.fields:
  Element<F> &${field};
% endfor
  // from the components of the array at a cell, read or written: a read of one that is written is const here
  template <class G, int N> ${fnqual} ${spec.accessor}(const Components<G, N> &v)
% for slot, field in list(enumerate(spec.fields)):
      ${":" if loop.first else ","} ${field}(elem(static_cast<F *>(v.at) + ${slot} * v.stride))
% endfor
  {}
% for s in spec.structs:
  ${fnqual} void store(const ${s.name} &s) const {
% for field in s.fields:
    ${field} = s.${field};
% endfor
  }
  ${fnqual} void load(${s.name} &s) const {
% for field in s.fields:
    s.${field} = ${field};
% endfor
  }
% endfor
  template <class G> ${fnqual} void copy(const ${spec.accessor}<G> &other) const {
% for field in spec.fields:
    ${field} = other.${field};
% endfor
  }
};
% endfor
} // namespace pg
