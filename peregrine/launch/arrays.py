import numpy as np

from peregrine.codegen.members import ScalarMember, ViewMember


class Broadcast:
    fullexts = ()

    @classmethod
    def of(cls, entry, spec):
        return cls()


class BaseSpec:
    dtype = None
    structs = comps = ()

    @property
    def ctype(self):
        return "fpdtype" if self.dtype is None else f"{self.dtype.name}_t"


class ArraySpec(BaseSpec):
    def __init__(self, name, kind, comps=(), *, axis=None, structs=None, exchanged=False, vecstart=None, nregs=1, dtype=None):
        self.name, self.kind, self.axis = name, kind, axis
        self.dtype = None if dtype is None else np.dtype(dtype)
        if self.dtype is not None and self.dtype.kind not in "iu":
            raise ValueError(f"{name}: an element type is the case's precision or an integer, not {self.dtype}")
        self.structs = tuple(structs or ())
        if self.structs:
            comps = len(self.fields)
        self.comps = (comps,) if isinstance(comps, int) else tuple(comps)
        # exchanged is True for every halo plane or the number of planes to trade
        self.exchanged, self.vecstart = exchanged, vecstart
        self.nregs = nregs

    @property
    def banks(self):
        return [self.name] + [f"{self.name}{n}" for n in range(1, self.nregs)]

    def blkrange(self, blk):
        return self.kind.of(blk, self)

    def shape(self, blk):
        return self.blkrange(blk).fullexts + self.comps

    def alloc(self, entry, backend, name=None):
        return backend.alloc(self.shape(entry), self.dtype, name=name or self.name)

    @property
    def fields(self):
        return tuple(f for s in self.structs for f in s.fields)

    def check(self, decl, expected):
        kind, axis = expected
        dims = () if self.structs else self.comps
        fits = issubclass(self.kind, kind) and axis in (None, self.axis)
        if not fits or decl.dims != dims:
            want = f"{kind.__name__}{list(decl.dims)} on axis {axis}"
            got = f"{self.kind.__name__}{list(self.comps)} on axis {self.axis}"
            raise ValueError(f"{decl.name} wants {want}; {self.name} is {got}")

    def member(self, decl, column):
        const = not decl.intent.writes
        return ViewMember(f"{decl.name}_", self, column, const)

    @property
    def accessor(self):
        return f"{self.name[0].upper()}{self.name[1:]}"


class ScalarSpec(BaseSpec):
    def __init__(self, scalar):
        self.scalar, self.name = scalar, scalar.name

    def check(self, decl, expected):
        if decl.dims:
            raise ValueError(f"{decl.name} is a scalar, declared bare")

    def member(self, decl, column):
        return ScalarMember(f"{decl.name}_", self)
