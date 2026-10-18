import ctypes
import re

import numpy as np

from peregrine.backend.abi import noffsets
from peregrine.codegen.grammar import Declaration
from peregrine.codegen.members import IndexMember, ParamMember, ShapeMember, ViewMember
from peregrine.launch.policy import BasePolicy, ListOffset


class Binding:
    def __init__(self, decl, member, local, offset):
        self.name, self.intent, self.offset = local, decl.intent, offset
        self.member = member

    def cxx(self, policy):
        expr = self.member.binding(self.offset, policy)
        spec = self.member.spec
        if spec.structs:
            const = "" if self.intent.writes else "const "
            view = f"pg::{spec.accessor}<{const}fpdtype>"
            return f"const {view} {self.name}{{{expr}}};"
        # a list is its accessor whatever the array's components
        if isinstance(self.offset, ListOffset):
            return f"const auto {self.name} = {expr};"
        if not spec.comps:
            if self.intent.writes:
                return f"{spec.ctype} &{self.name} = {expr};"
            return f"const {spec.ctype} {self.name} = {expr};"
        return f"const auto {self.name} = {expr};"


class Kernel:
    def __init__(
        self,
        provider,
        template,
        *,
        name=None,
        axis=None,
        options=None,
        columns=None,
        externs=None,
        **arguments,
    ):
        self.template = template
        # None takes the tag's name at declaration; given, it picks one tag of several
        self.name = name
        self.axis = axis
        # the kernel's own template arguments: what its template branches on
        self.options = dict(options or {})
        # the columns it binds beyond the case's arrays, a collection's own
        self.columns = dict(columns or {})
        # declarations the caller adds to the tag's own: what its options' expressions read
        self.externs = dict(externs or {})
        # what each declared name is bound to: an array or scalar of the case by name, or a number
        self.binds = arguments
        # the provider that declared it, and the arguments it was declared against
        self.provider = self.args = None
        self.spec = self.policy = None
        self.members, self.bindings = [], []
        self.struct = None
        self._text = self._fn = None
        self.declare(provider)

    def __repr__(self):
        return f"<Kernel {self}>"

    def __str__(self):
        stem = self.template.rsplit(".", 1)[1]
        axis = f" {'ijk'[self.axis]}" if self.axis is not None else ""
        name = stem if self.name == stem else f"{stem}.{self.name}"
        return name + axis

    ###########################################################################
    # Declaring for a case
    ###########################################################################
    def declare(self, provider):
        self.provider = provider
        specs = provider.declare(self, self.columns)
        # a kernel takes its name from its tag, when the template has one
        if self.name is None:
            self.name = next(iter(specs))
        self.spec = specs[self.name]
        self.policy = BasePolicy.named(self.spec.over, self.axis)
        self._resolve()
        provider.admit(self)

    @property
    def entry(self):
        return "reduction" if self.reduce else "launch"

    @property
    def symbol(self):
        return re.sub(r"\W+", "_", str(self))

    # the widest stencil among a case's kernels sizes its halo
    @property
    def stencil(self):
        return max((b.offset.reach for b in self.bindings), default=0)

    @property
    def reduce(self):
        return self.spec.reduce

    @property
    def ncomps(self):
        return self.policy.ncomps(self)

    @property
    def groups(self):
        return [[self]]

    def decls(self):
        decls = self.spec.decls.items()
        return [Declaration.parse(n, t, self.policy) for n, t in decls]

    def _member(self, decl):
        column = self.binds[decl.name]
        spec = self.args["arrays"][column]
        for _, offset in decl.offsets:
            spec.check(decl, self.policy.expects(offset))
        return spec.member(decl, column)

    def _resolve(self):
        self.bindings, self.members = [], []
        for d in self.decls():
            member = self._member(d)
            self.members.append(member)
            for local, offset in d.offsets:
                self.bindings.append(Binding(d, member, local, offset))
        # the carried offsets, in declaration order, for the ranges to supply
        offsets = [b.offset for b in self.bindings if b.offset.carried]
        self.policy.carried = list({o.key: o for o in offsets}.values())
        tpl, limit = self.template, noffsets
        if len(self.policy.carried) > limit:
            raise ValueError(f"{tpl} names more than {limit} carried offsets")
        # an indexed kernel's views each get a shape member and all share one index member
        if self.policy.index is not None:
            shapes = []
            for m in self.members:
                if isinstance(m, ViewMember):
                    shape = ShapeMember(f"{m.cname}shape_", m.cname)
                    m.shape = shape.cname
                    shapes.append(shape)
            self.members += [*shapes, IndexMember("at_")]
        # the kernel's own numbers, bound by name, in the case's precision
        fpctype = self.provider.backend.fpctype
        self.members += [ParamMember(n, fpctype) for n in self.spec.params]
        fields = [(m.cname, m.twin) for m in self.members]
        namespace = {"_fields_": fields}
        self.struct = type(self.name, (ctypes.Structure,), namespace)

    ###########################################################################
    # Rendering, compiling, calling
    ###########################################################################
    @property
    def text(self):
        if self._text is None:
            self._text = self.provider.text(self)
        return self._text

    @property
    def fn(self):
        if self._fn is None:
            self.provider.build([self])
        return self._fn

    def bind(self, function):
        self._fn = function

    def functor(self, table):
        filled = self.struct()
        for m in self.members:
            m.fill(filled, table, self)
        return filled

    def launches(self, tiling):
        if not tiling.tiles:
            return False
        if tiling.tilekind != self.policy.tilekind:
            raise TypeError(f"{self} runs over {self.policy.tilekind}, this tiling is of {tiling.tilekind}")
        return True

    def __call__(self, table, tiling):
        out = None
        if self.reduce:
            out = np.full(self.reduce.count, self.reduce.identity, self.provider.backend.fpdtype)
        if self.launches(tiling):
            self.fn(self.functor(table), tiling, out)
        return out

    def node(self, graph, deps, table, tiling):
        if not self.launches(tiling):
            return None
        return self.fn.node(graph, deps, self.functor(table), tiling)


# kernels of one group read and write nothing of each other's, so no order holds between them
class KernelGroup:
    def __init__(self, kernels, name=None):
        self.kernels = list(kernels)
        self.name = name or str(self.kernels[0])

    def __str__(self):
        return self.name

    @property
    def groups(self):
        return [self.kernels]
