import copy
import ctypes
import re

from mako.runtime import capture

from peregrine.backend import abi
from peregrine.backend.base.templates import Templates
from peregrine.misc import subclass_where, merge_disjoint
from peregrine.codegen.constants import BaseConstants, LiteralConstants
from peregrine.codegen.grammar import Declaration, Names, Reduction, Substitution


class BaseRender:
    def __init__(self, args, kernel):
        self.args, self.kernel = args, kernel

    @property
    def context(self):
        return merge_disjoint(self.args, {"kernel": self.kernel}, self.kernel.options)


class KernelSpec:
    def __init__(self, name, over, params="", decls=None, reduce=None):
        self.name, self.over = name, over
        self.params, self.decls = Names(params), dict(decls or {})
        self.reduce = Reduction(reduce) if reduce else None
        self.body = None


class Macro:
    def __init__(self, name, params, body, origin, decls=None):
        self.name, self.params = name, Names(params)
        self.body, self.origin = body, origin
        self.decls = dict(decls or {})
        self.subst = Substitution(self.params)

    def expand(self, context, args, scoped=True):
        body = capture(context, self.body)
        text = self.subst.apply(body, args).strip("\n")
        if not scoped:
            return f"// {self.origin}: {self.name}\n{text}"
        return "\n".join([f"{{  // {self.origin}: {self.name}", text, "}"])


class KernelRender(BaseRender):
    def __init__(self, args, kernel):
        super().__init__(args, kernel)
        # what every tag declared, by name
        self.specs = {}
        # the macros the included files defined, by name
        self.macros = {}
        # declared by the included files, shared by every kernel
        self.constants = {}
        # the tag whose body is being captured
        self.current = None

    def declare(self, spec):
        if spec.name in self.specs:
            tpl = self.kernel.template
            raise ValueError(f"{tpl} holds two kernels named {spec.name!r}")
        self.specs[spec.name] = spec
        spec.constants = self.constants
        self.current = spec

    def define(self, macro):
        held = self.macros.get(macro.name)
        # a file included twice defines its macros again; only another file defining the name is refused
        if held and held.origin != macro.origin:
            tpl = self.kernel.template
            both = f"{macro.name!r}, from {held.origin} and {macro.origin}"
            raise ValueError(f"{tpl}: two macros named {both}")
        self.macros[macro.name] = macro

    def hold(self, values, origin):
        for name, value in values.items():
            # the same file again is fine, another file is not
            if self.constants.get(name, value) is not value:
                raise ValueError(f"{origin}: {name!r} is already declared")
            self.constants[name] = value

    def alias(self, context, name, file, under=None):
        aside = KernelRender(self.args, self.kernel)
        scope = context._locals({"_render": aside})
        # Mako caches a template's namespaces on the context, each bound to the context it was made in, and a copy shares the cache: the aside takes its own, or a file this render runs after the aside ran it would call the aside's tags
        scope.namespaces = {}
        capture(context, Templates.get(file).render_context, scope)
        self.constants.update(aside.constants)
        if name in aside.specs:
            spec = aside.specs[name]
            spec.name = under or name
            self.declare(spec)
        else:
            macro = copy.copy(aside.macros[name])
            macro.name = under or name
            self.define(macro)

    def expand(self, context, name, args, scoped=True):
        macro = self.macros[name]
        decls = self.current.decls
        for key, text in macro.decls.items():
            decls[key] = Declaration.merge(key, decls.get(key), text)
        return macro.expand(context, args, scoped)

    def run(self):
        ctx = self.context
        Templates.get(self.kernel.template).render(_render=self, **ctx)
        return self.specs


class Unit(BaseRender):
    # a decimal floating literal, not already suffixed and not the mantissa of a hex one
    dotted = r"(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?"
    exponent = r"\d+[eE][+-]?\d+"
    floatlit = re.compile(rf"(?<![\w.])({dotted}|{exponent})(?![\w.])")

    # an unsuffixed decimal literal is a double and promotes the arithmetic around it
    @classmethod
    def single(cls, text):
        return cls.floatlit.sub(r"\1f", text)

    def text(self):
        kernel, policy = self.kernel, self.kernel.policy
        double = self.args["fpdtype"] == "double"
        body = kernel.spec.body if double else self.single(kernel.spec.body)
        lines = body.strip("\n").splitlines()
        body = "\n".join(("  " + line).rstrip() for line in lines)
        signature = [policy.signature]
        if kernel.reduce:
            signature.append(f"fpdtype *{kernel.reduce.values}")
        context = self.context
        structs = Templates.get("peregrine.codegen.kernels.structs").render(**context)
        fnqual, memqual = self.args["fnqual"], self.args["memqual"]
        placed = subclass_where(BaseConstants, name=self.args["constform"])(self.args["constants"], fnqual, memqual)
        # a template's own numbers stay literal
        declared = LiteralConstants(kernel.spec.constants, fnqual, memqual)
        text = Templates.get(f"{Templates.base}.unit").render(
            base=Templates.base,
            abi=abi,
            policy=policy,
            body=body,
            structdefs=structs.strip("\n"),
            accessors=placed.render(body) + declared.render(body),
            signature=", ".join(s for s in signature if s),
            size=ctypes.sizeof(kernel.struct),
            **context,
        )
        return text.strip() + "\n"
