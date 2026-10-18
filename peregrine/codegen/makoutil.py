# the pg namespace of every template: Mako passes a module namespace's functions the context first
from mako.runtime import capture, supports_caller

from peregrine.misc import merge_disjoint
from peregrine.codegen.render import KernelSpec, Macro


@supports_caller
def kernel(context, name, over, params="", decls=None, reduce=None, **declarations):
    decls = merge_disjoint(decls or {}, declarations)
    # what the caller declares beyond the tag, as the kernel's externs; a tag may say the same, never otherwise
    for extern, text in context["kernel"].externs.items():
        if decls.get(extern, text) != text:
            what = f"{name}: {extern} declared {decls[extern]!r} by the tag"
            raise ValueError(f"{what} and {text!r} by the caller")
        decls[extern] = text
    spec = KernelSpec(name, over, params, decls, reduce)
    context["_render"].declare(spec)
    spec.body = capture(context, context["caller"].body)
    return ""


@supports_caller
def macro(context, name, params, **declarations):
    body, origin = context["caller"].body, context["local"].uri
    macro = Macro(name, params, body, origin, declarations)
    context["_render"].define(macro)
    return ""


def alias(context, name, file, under=None):
    context["_render"].alias(context, name, file, under)
    return ""


def expand(context, name, args="", scoped=True):
    return context["_render"].expand(context, name, args, scoped)


# the numbers a file's macros read, literal whatever the constform
def constants(context, **values):
    context["_render"].hold(values, context["local"].uri)
    return ""


def dot(context, a, b=None, **runs):
    for k, v in runs.items():
        if isinstance(v, tuple):
            runs[k] = range(*v)
        elif isinstance(v, int):
            runs[k] = range(v)
    terms = []
    for values in zip(*runs.values()):
        at = dict(zip(runs, values))
        terms.append(f"({a.format(**at)}) * ({(b or a).format(**at)})")
    return "(" + " + ".join(terms) + ")"
