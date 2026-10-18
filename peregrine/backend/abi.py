import ctypes


class AbiStruct:
    # the ctypes of the C types a field may have; a pointer is an address whatever it points to
    ctypes_of = {"int": ctypes.c_int, "long": ctypes.c_long, "fpdtype": None}

    def __init__(self, name, fields):
        self.name, self.fields = name, tuple(fields)
        twins = []
        for field in self.fields:
            fname, ctype, *lengths = field
            twin = ctypes.c_void_p if ctype.endswith("*") else self.ctypes_of[ctype]
            for length in reversed(lengths):
                twin = twin * length
            twins.append((fname, twin))
        self.twin = type(name, (ctypes.Structure,), {"_fields_": twins})

    def cxx(self, indent=""):
        lines = []
        for fname, ctype, *lengths in self.fields:
            dims = "".join(f"[{n}]" for n in lengths)
            lines.append(f"{ctype}{'' if ctype.endswith('*') else ' '}{fname}{dims};")
        return f"\n{indent}".join(lines)

    @property
    def size(self):
        return ctypes.sizeof(self.twin)


# how many offsets a range carries
noffsets = 4

structs = {
    s.name: s
    for s in (
        # all a kernel needs to know about one array, as Python holds it: where, how big, and its strides in elements
        AbiStruct("ArrInfo", [("data", "void *"), ("rank", "int"), ("extent", "int", 5), ("stride", "long", 5)]),
        # one range of cells a launch does: a start and an extent per axis, the components, the item count, the entry, the offsets it carries
        AbiStruct("Range", [("start", "int", 3), ("extent", "int", 4), ("n", "int"), ("entry", "int"), ("offset", "int", noffsets, 3)]),
        # one launch over a table: the range of each tile, the first tile and the items of each range, the ranges, and the items a tile is
        AbiStruct("Tiling", [("range", "const int *"), ("first", "const int *"), ("items", "const int *"), ("ranges", "const Range *"), ("count", "int"), ("tiles", "int"), ("tile", "int")]),
        # a view: the arrinfos, then the entry's data the launch pins
        AbiStruct("View", [("arrinfos", "const ArrInfo *"), ("data", "T *")]),
        # the shape a view is indexed in: its extents and its strides in elements
        AbiStruct("Shape", [("n", "int", 3), ("s", "long", 5)]),
        # where a thread stands: the entry, the cell, the range it is of, and the slot a listed launch pinned it from
        AbiStruct("Index", [("entry", "int"), ("i", "int"), ("j", "int"), ("k", "int"), ("range", "const Range *"), ("slot", "int")]),
        # one scalar where the kernels run, read through its address
        AbiStruct("Scalar", [("data", "const fpdtype *")]),
    )
}

ArrInfo, Range, Tiling = structs["ArrInfo"].twin, structs["Range"].twin, structs["Tiling"].twin
View, Shape, Index, Scalar = structs["View"].twin, structs["Shape"].twin, structs["Index"].twin, structs["Scalar"].twin

# the arrayInfo of an array an entry does not have: a kernel never reads it
noarray = ArrInfo(None, 0, (ctypes.c_int * 5)(), (ctypes.c_long * 5)())
