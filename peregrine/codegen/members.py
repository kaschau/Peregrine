from peregrine.backend.abi import Index, Scalar, Shape, View


class BaseMember:
    twin = None
    # the table column it holds, for one that holds one
    column = None

    def __init__(self, cname, const=True):
        self.cname, self.const = cname, const

    def decl(self):
        raise NotImplementedError

    def pin(self):
        return ""

    def fill(self, filled, table, kernel):
        raise NotImplementedError


class ViewMember(BaseMember):
    twin = View

    def __init__(self, cname, spec, column, const):
        super().__init__(cname, const)
        self.spec, self.column = spec, column
        # the name of the ShapeMember this view is indexed in, set by the kernel
        self.shape = None

    def decl(self):
        dims = "".join(f", {d}" for d in self.spec.comps)
        const = "const " if self.const else ""
        return f"pg::View<{const}{self.spec.ctype}{dims}> {self.cname};"

    def fill(self, filled, table, kernel):
        view = self.twin(arrinfos=table.pointer(self.column))
        setattr(filled, self.cname, view)

    def pin(self):
        return f"{self.cname}.pin(at);"

    def binding(self, offset, policy):
        if policy.index is None:
            return f"{self.cname}.flat()"
        return offset.binding(self)


class ShapeMember(BaseMember):
    twin = Shape

    def __init__(self, cname, source):
        super().__init__(cname)
        self.source = source

    def decl(self):
        return f"pg::Shape {self.cname};"

    # the launch pins it
    def fill(self, filled, table, kernel):
        pass

    def pin(self):
        return f"{self.cname}.pin({self.source}.arrinfos, at.e);"


class IndexMember(BaseMember):
    twin = Index

    def decl(self):
        return f"pg::Index {self.cname};"

    # the launch pins it
    def fill(self, filled, table, kernel):
        pass


class ScalarMember(BaseMember):
    twin = Scalar

    def __init__(self, cname, spec):
        super().__init__(cname)
        self.spec = spec

    def decl(self):
        return f"pg::Scalar {self.cname};"

    def fill(self, filled, table, kernel):
        setattr(filled, self.cname, Scalar(self.spec.scalar.ptr))

    def binding(self, offset, policy):
        return f"{self.cname}()"


class ParamMember(BaseMember):
    def __init__(self, cname, fpctype):
        super().__init__(cname)
        self.twin = fpctype

    def decl(self):
        return f"fpdtype {self.cname};"

    def fill(self, filled, table, kernel):
        setattr(filled, self.cname, kernel.binds[self.cname])
