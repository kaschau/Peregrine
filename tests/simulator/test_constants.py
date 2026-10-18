import re

import cantera as ct
import numpy as np
import pytest

import peregrine as pg
from peregrine.config import Config
from tests.chemint.cases import at_cell, cell, randomstate
from tests.gases import same
from tests.gate import cases

reacting = ["navstokes rk3 CH4_O2 explicit", "navstokes rk3 CH4_O2 sub-stepped"]
mechanisms = ["CH4_O2_FFCMY.yaml", "C2H4_Air_Skeletal.yaml", "GRI30.yaml"]


def formed(backend, constform):
    def make(config):
        made = backend(config)
        made.constform = constform
        return made

    return make


def texts(mb):
    return [k.text for g in mb.graphs.values() for k in g.kernels]


def test_the_config_names_the_form(backend):
    config = Config()
    config[f"backend-{backend.name}"]["constform"] = "memory"
    made = pg.get_backend(backend.name, config)
    assert made.constform == "memory"
    made.stop()


def test_the_default_form_is_the_backends(backend):
    # the form is the config's
    mb = cases.build(backend, **cases.cases["navstokes rk3 CH4_O2"])
    defined = any(f"{mb.backend.memqual} fpdtype " in t for t in texts(mb))
    assert defined == (mb.backend.constform == "memory")


@pytest.mark.parametrize("name", reacting)
def test_the_forms_give_the_same_answer(backend, name):
    literal = cases.build(backend, **cases.cases[name])
    memory = cases.build(formed(backend, "memory"), **cases.cases[name])
    for a, b in zip(cases.rhs(literal).values(), cases.rhs(memory).values()):
        assert same(literal, a, b)
    cases.step(literal)
    cases.step(memory)
    for x, y in zip(literal.blocks, memory.blocks):
        for name in (*literal.regs, "state"):
            assert same(literal, getattr(x, name).get(), getattr(y, name).get())
    # the case's lookups are definitions, a template's own stay literal
    assert any(f"{memory.backend.memqual} fpdtype " in t for t in texts(memory))
    literal = {m for t in texts(memory) for m in re.findall(r"fpdtype (\w+)\(const int i0\) \{ static constexpr", t)}
    assert literal == {"logtab", "rcptab", "exptab"}


def test_the_forms_agree_in_the_log_form(backend):
    # single precision renders the log form
    p, T, Y = randomstate(12, np.random.default_rng(11))
    omega = {}
    for constform in ("literal", "memory"):
        mb = cell(formed(backend, constform), "CH4_O2_FFCMY.yaml", "coupled", p, T, Y, precision="single")
        assert ", product form" not in mb.kernels["rates"](Q=mb.regs[0]).text
        omega[constform] = at_cell(mb, "omega")
    assert np.isfinite(omega["literal"]).all() and same(mb, omega["literal"], omega["memory"])


@pytest.mark.parametrize("mechanism", mechanisms)
@pytest.mark.parametrize("kind", ["coupled", "sub-stepped"])
def test_a_units_memory_fits_the_constant_bank(backend, mechanism, kind):
    # CUDA caps a unit's __constant__ at 64 KB
    p, T, Y = randomstate(ct.Solution(mechanism).n_species, np.random.default_rng(3))
    mb = cell(formed(backend, "memory"), mechanism, kind, p, T, Y)
    memqual, itemsize = re.escape(mb.backend.memqual), mb.backend.fpdtype.itemsize
    for text in texts(mb):
        lookups = sum(int(n) for n in re.findall(rf"{memqual} fpdtype \w+_lookup\[(\d+)\]", text))
        scalars = len(re.findall(rf"{memqual} fpdtype \w+ = ", text))
        assert (lookups + scalars) * itemsize < 65536
