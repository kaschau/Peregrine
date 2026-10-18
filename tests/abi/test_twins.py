import ctypes

import pytest

from peregrine.backend import abi

twins = {
    # struct, size in bytes, (field, offset) pairs
    abi.ArrInfo: (72, (("data", 0), ("rank", 8), ("extent", 12), ("stride", 32))),
    abi.Range: (84, (("start", 0), ("extent", 12), ("n", 28), ("entry", 32), ("offset", 36))),
    abi.Tiling: (48, (("range", 0), ("first", 8), ("items", 16), ("ranges", 24), ("count", 32), ("tiles", 36), ("tile", 40))),
    # the functor's members, as views.mako asserts them
    abi.View: (16, (("arrinfos", 0), ("data", 8))),
    abi.Shape: (56, (("n", 0), ("s", 16))),
    abi.Index: (32, (("entry", 0), ("i", 4), ("j", 8), ("k", 12), ("range", 16), ("slot", 24))),
    abi.Scalar: (8, (("data", 0),)),
}


@pytest.mark.parametrize("twin", list(twins), ids=lambda t: t.__name__)
def test_twin(twin):
    size, offsets = twins[twin]
    assert ctypes.sizeof(twin) == size
    for field, offset in offsets:
        assert getattr(twin, field).offset == offset


def test_a_range_carries_four_offsets():
    # the count sizes the struct on both sides
    assert abi.noffsets == 4


def test_the_fields_render_as_the_twin_is_laid_out():
    assert abi.structs["ArrInfo"].cxx() == "void *data;\nint rank;\nint extent[5];\nlong stride[5];"
    assert abi.structs["Tiling"].cxx("  ").splitlines()[3] == "  const Range *ranges;"
    assert abi.structs["Range"].cxx().splitlines()[4] == "int offset[4][3];"
    assert abi.structs["Index"].size == ctypes.sizeof(abi.Index)
