import ast

import numpy as np

from peregrine.config import ConfigError
from peregrine.launch.arrays import ArraySpec, Broadcast
from peregrine.launch.kernel import Kernel


class BaseBc:
    # what this kind of boundary is called, which the config's entry names
    kind = None
    # the variables an entry of this kind gives, in order
    variables = ()

    def __init__(self, name, given, sim):
        self.name, self.sim = name, sim
        privars = sim.iovars.privars
        self.values = BcValues(self, given, privars) if self.variables else None

    @classmethod
    def varnames(cls, privars):
        return cls.variables

    @classmethod
    def template(cls):
        return f"{cls.__module__.rsplit('.', 1)[0]}.kernels.bcs.{cls.kind}"

    @property
    def columns(self):
        return {"outward": ArraySpec("outward", Broadcast, 1)}

    def start(self, tme, interfaces):
        for interface in interfaces:
            sign = -1.0 if interface.is_low else 1.0
            interface.outward.set(np.full(interface.outward.shape, sign))

    # writes its interfaces' arrays in place: the device graphs are built over them
    def update(self):
        pass

    def options(self):
        return self.values.options if self.values else None

    def kernels(self, hook, Q):
        return getattr(self, hook)(Q)

    def _kernels(self, hook, faces=False, **arguments):
        columns = self.columns
        own = dict(options=self.options(), columns=columns)
        mine = {name: name for name in columns}
        tpl = self.template()
        externs = self.values.externs(faces) if self.values else {}
        kernels = []
        for d, a in enumerate("ijk"):
            axis = {k: v.format(a=a) if isinstance(v, str) else v for k, v in arguments.items()}
            kernels.append(Kernel(self.sim.provider, tpl, name=hook, axis=d, S=f"{a}S", xf=f"{a}xc", tme="tme", externs=externs, **own, **mine, **axis))
        return kernels


class BcValues:
    def __init__(self, kind, given, privars):
        self.given = given
        self.variables = kind.variables
        self.varnames = kind.varnames(privars)
        values, self.uses = {}, set()
        for name in self.varnames:
            if name in given:
                values[name] = self._value(given[name])
            elif name in kind.variables:
                what = f"{kind.kind} reads {name}"
                msg = f"{what}, which the entry does not give"
                raise ConfigError(f"bcvals.{kind.name}", msg)
            else:
                values[name] = "0.0"
        self.values = values

    def _value(self, value):
        if isinstance(value, str):
            expression = Expression(value)
            self.uses |= expression.uses
            return expression.cxx
        return repr(float(value))

    def externs(self, faces=False):
        externs = {}
        if self.uses & {"x", "y", "z"}:
            externs["xf"] = "in [3]" if faces else "in [3] at intface"
        if "t" in self.uses:
            externs["tme"] = "in"
        return externs

    @property
    def options(self):
        options = {v: self.values[v] for v in self.variables}
        species = [n for n in self.varnames if n not in self.variables]
        options["Y"] = [self.values[s] for s in species]
        return options


class Expression:
    names = {"x": "xf[0]", "y": "xf[1]", "z": "xf[2]", "t": "tme"}
    functions = dict(
        sin="sin", cos="cos", tan="tan", exp="exp", log="log", sqrt="sqrt",
        tanh="tanh", abs="fabs", min="fmin", max="fmax", pow="pow",
    )
    operators = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}

    def __init__(self, text):
        self.text = text
        # which of the position and the clock the text reads
        self.uses = set()
        self.cxx = self._cxx(ast.parse(text, mode="eval").body)

    def _cxx(self, node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return repr(float(node.value))
        if isinstance(node, ast.Name):
            self.uses.add(node.id)
            return self._known(node.id, self.names)
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.USub, ast.UAdd):
            sign = "-" if isinstance(node.op, ast.USub) else ""
            return f"({sign}{self._cxx(node.operand)})"
        if isinstance(node, ast.BinOp):
            a, b = self._cxx(node.left), self._cxx(node.right)
            if isinstance(node.op, ast.Pow):
                return f"pow({a}, {b})"
            return f"({a} {self._known(type(node.op), self.operators)} {b})"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = self._known(node.func.id, self.functions)
            return f"{fn}({', '.join(self._cxx(a) for a in node.args)})"
        what = type(node).__name__
        raise ValueError(f"{self.text!r}: {what} is no part of an expression a kernel evaluates")

    def _known(self, key, table):
        try:
            return table[key]
        except KeyError:
            known = ", ".join(getattr(k, "__name__", k) for k in table)
            name = getattr(key, "__name__", key)
            raise ValueError(f"{self.text!r}: {name!r}; a kernel knows {known}") from None
