import re

import numpy as np


# a body's constants as the accessors it calls, in one form
class BaseConstants:
    name = None

    def __init__(self, constants, fnqual, memqual):
        self.constants, self.fnqual, self.memqual = constants, fnqual, memqual

    # a bare double narrows inside a float array's braces
    @staticmethod
    def literal(x):
        return f"fpdtype({float(x).hex()})"

    @classmethod
    def literals(cls, values):
        return "{" + ", ".join(cls.literal(x) for x in np.asarray(values).ravel()) + "}"

    def names(self, body):
        words = set(re.findall(r"\w+", body))
        return [c for c in self.constants if words & {c, c + "_terms"}]

    def render(self, body):
        return [self.accessor(name, np.asarray(self.constants[name])) for name in sorted(self.names(body))]

    # a scalar is a variable, a lookup a function of one index per axis; NAME_terms its last extent
    def accessor(self, name, value):
        if value.ndim == 0:
            return self.scalar(name, value)
        axes = [f"i{d}" for d in range(value.ndim)]
        offsets = [int(np.prod(value.shape[d + 1 :])) for d in range(value.ndim)]
        index = " + ".join(f"{ax} * {off}" if off > 1 else ax for ax, off in zip(axes, offsets))
        head = f"{self.fnqual} fpdtype {name}({', '.join(f'const int {ax}' for ax in axes)})"
        return f"constexpr int {name}_terms = {value.shape[-1]};\n{self.lookup(name, head, value, index)}"

    def scalar(self, name, value):
        raise NotImplementedError

    # the values and the function, head, reading them at index
    def lookup(self, name, head, values, index):
        raise NotImplementedError


class LiteralConstants(BaseConstants):
    name = "literal"

    def scalar(self, name, value):
        return f"constexpr fpdtype {name} = {self.literal(value)};"

    # folds under a literal index, a global load under a run-time one
    def lookup(self, name, head, values, index):
        return f"{head} {{ static constexpr fpdtype t[{values.size}] = {self.literals(values)}; return t[{index}]; }}"


class MemoryConstants(BaseConstants):
    name = "memory"

    def scalar(self, name, value):
        return f"{self.memqual} fpdtype {name} = {self.literal(value)};"

    def lookup(self, name, head, values, index):
        return f"{self.memqual} fpdtype {name}_lookup[{values.size}] = {self.literals(values)};\n{head} {{ return {name}_lookup[{index}]; }}"
