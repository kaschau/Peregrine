// rendered by peregrine from ${kernel.template} for one case; do not edit
using fpdtype = ${fpdtype};
<%include file="${base}.abi"/>
<%include file="${backend.templates}.head"/>
<%include file="${base}.views"/>
<%include file="${base}.policy"/>
${structdefs}
% if accessors:
// the case's constants
% for line in accessors:
${line}
% endfor
% endif
struct ${kernel.name} {
% if policy.carried:
  // the offsets the ranges carry, named behind an interface
  enum {
% for n, offset in list(enumerate(policy.carried)):
    ${offset.key} = ${n}${"" if loop.last else ","}
% endfor
  };
% endif
% for member in kernel.members:
  ${member.decl()}
% endfor
  // per entry the views' data and each view's shape; per item the one index
  ${fnqual} void pin(const pg::Entry &at) {
% for member in kernel.members:
% if member.pin():
    ${member.pin()}
% endif
% endfor
  }
% if policy.index is not None:
  // the thread's index, and whether the item is one to run
  ${fnqual} bool pin(const ${policy.index} &at) {
    at_ = at;
% for line in policy.runs(kernel):
    ${line}
% endfor
  }
% endif
  ${fnqual} void operator()(${signature}) const {
% for binding in kernel.bindings:
    ${binding.cxx(policy)}
% endfor
    // ${kernel.template}
${body}
  }
};
static_assert(sizeof(${kernel.name}) == ${size}, "laid out as the python twin");
<%def name="item(values='', indent='')">\
${indent}const auto at = ${policy.cxx}::decode(t.ranges[r.r], item, r.e);
% if policy.index is not None:
${indent}${kernel.name} q = p;
${indent}if (q.pin(at))
${indent}  q(${values});
% else:
${indent}p(at);
% endif
</%def>\
<%include file="${backend.entries}.${kernel.entry}" args="item=item"/>
