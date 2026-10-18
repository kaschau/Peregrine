import re

import numpy as np

from peregrine.misc import subclass_where


class Names(tuple):
    def __new__(cls, text):
        return super().__new__(cls, filter(None, re.split(r"[\s,]+", text or "")))


class BaseIntent:
    name = None
    reads = False
    writes = False


class InIntent(BaseIntent):
    name = "in"
    reads = True


class OutIntent(BaseIntent):
    name = "out"
    writes = True


class InOutIntent(BaseIntent):
    name = "inout"
    reads = writes = True


class Reduction:
    def __init__(self, text):
        values, op, count = text.split()
        self.values, self.count = values, int(count.strip("[]"))
        # the C fold and numpy's are spelled alike, fmax and fmin
        self.cfunc = f"f{op}"
        self.ufunc = getattr(np, self.cfunc)
        self.sign = "-" if op == "max" else ""
        self.identity = float(f"{self.sign}inf")


class Declaration:
    def __init__(self, name, intent, dims, offsets):
        self.name, self.intent, self.dims = name, intent, dims
        # (local, offset) per place read
        self.offsets = offsets

    pattern = re.compile(r"\s*(\w+)\s*(\[[^\]]*\])?\s*(?:at\s+(.*?))?\s*")

    @classmethod
    def split(cls, name, text):
        match = cls.pattern.fullmatch(text)
        if match is None:
            form = "an intent, components in brackets, then 'at'"
            raise ValueError(f"{name}={text!r}: expected {form}")
        intent, dims, at = match.groups()
        dims = tuple(int(d) for d in re.findall(r"\d+", dims or ""))
        return intent, dims, [w.strip() for w in (at or "").split(",")]

    @classmethod
    def merge(cls, name, held, text):
        if held is None:
            return text
        intent, dims, words = cls.split(name, held)
        other = cls.split(name, text)
        if other[:2] != (intent, dims):
            raise ValueError(f"{name}: declared {held!r} and {text!r}")
        words += [w for w in other[2] if w not in words]
        dims = f" [{', '.join(map(str, dims))}]" if dims else ""
        at = f" at {', '.join(words)}" if words else ""
        return f"{intent}{dims}{at}"

    @classmethod
    def parse(cls, name, text, policy):
        intent, dims, words = cls.split(name, text)
        # the local for each place read: the name and the offset's word, a sign spelt out, QL, QpI, QmI
        locals_ = [name + w.replace("+", "p").replace("-", "m") for w in words]
        names = [name] if len(words) == 1 else locals_
        offsets = list(zip(names, map(policy.offset, words)))
        intent = subclass_where(BaseIntent, name=intent)
        return cls(name, intent, dims, offsets)


class Substitution:
    identifier = re.compile(r"[A-Za-z_]\w*(\[[^\]]*\])*|[0-9.]+")

    def __init__(self, params):
        self.params = tuple(params)
        names = "|".join(map(re.escape, self.params))
        # never after a member access: s.x or p->x is a member of that name, not the parameter
        self.pattern = re.compile(rf"(?<![\w.])(?<!->)\b({names})\b")

    @staticmethod
    def split(text):
        args, depth, start = [], 0, 0
        for i, ch in enumerate(text):
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif ch == "," and depth == 0:
                args.append(text[start:i].strip())
                start = i + 1
        tail = text[start:].strip()
        if tail:
            args.append(tail)
        return args

    def apply(self, text, args):
        args = self.split(args)
        if len(self.params) != len(args):
            want = f"{len(self.params)} arguments {self.params}"
            raise ValueError(f"expected {want}, got {len(args)} {args}")
        words = [a if self.identifier.fullmatch(a) else f"({a})" for a in args]
        table = dict(zip(self.params, words))
        if not self.params:
            return text
        return self.pattern.sub(lambda m: table[m.group(1)], text)
