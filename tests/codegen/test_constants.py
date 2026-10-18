import numpy as np
import pytest

from peregrine.codegen.constants import BaseConstants
from peregrine.misc import subclass_where

constants = {"Ru": np.array(8314.46261815324), "MW": np.array([2.0, 32.0]), "poly": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
body = "Ru * MW(n) + poly(n, m) * poly_terms"
mw = "{fpdtype(0x1.0000000000000p+1), fpdtype(0x1.0000000000000p+5)}"
poly = "{fpdtype(0x1.0000000000000p+0), fpdtype(0x1.0000000000000p+1), fpdtype(0x1.8000000000000p+1), fpdtype(0x1.0000000000000p+2), fpdtype(0x1.4000000000000p+2), fpdtype(0x1.8000000000000p+2)}"


def rendered(constform, fnqual="inline", memqual="static const", text=body):
    return subclass_where(BaseConstants, name=constform)(constants, fnqual, memqual).render(text)


def test_the_literal_form_writes_the_values_into_the_code():
    assert rendered("literal") == [
        f"constexpr int MW_terms = 2;\ninline fpdtype MW(const int i0) {{ static constexpr fpdtype t[2] = {mw}; return t[i0]; }}",
        "constexpr fpdtype Ru = fpdtype(0x1.03d3b3712575ap+13);",
        f"constexpr int poly_terms = 3;\ninline fpdtype poly(const int i0, const int i1) {{ static constexpr fpdtype t[6] = {poly}; return t[i0 * 3 + i1]; }}",
    ]


def test_the_memory_form_keeps_the_values_where_the_backend_says():
    assert rendered("memory", "__device__ inline", "__constant__") == [
        f"constexpr int MW_terms = 2;\n__constant__ fpdtype MW_lookup[2] = {mw};\n__device__ inline fpdtype MW(const int i0) {{ return MW_lookup[i0]; }}",
        "__constant__ fpdtype Ru = fpdtype(0x1.03d3b3712575ap+13);",
        f"constexpr int poly_terms = 3;\n__constant__ fpdtype poly_lookup[6] = {poly};\n__device__ inline fpdtype poly(const int i0, const int i1) {{ return poly_lookup[i0 * 3 + i1]; }}",
    ]


def test_a_body_names_what_it_renders():
    assert rendered("literal", text="nothing here") == []
    assert [line.split("\n")[0] for line in rendered("literal", text="poly_terms")] == ["constexpr int poly_terms = 3;"]


def test_an_unknown_form_is_refused():
    with pytest.raises(KeyError):
        rendered("register")
