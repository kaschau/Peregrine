import numpy as np
import pytest

from peregrine.config import ConfigError

from tests.chemint.cases import at_cell, cell, dQ, interior, mechanism, stoichiometric


def reactants(backend, kind, tolerance, dt=1e-6):
    return cell(backend, mechanism, kind, 10e5, 400.0, stoichiometric(), dt=dt, maxsubsteps=200, equilibrium_tolerance=tolerance)


# ignited products never settle: at the explicit scheme's fixed point each substep throws the fast radicals off their quasi-steady values, so their rates are large at a step's start
def products(backend, tolerance, relax=2, dt=1e-6):
    mb = cell(backend, mechanism, "sub-stepped", 10e5, 2500.0, stoichiometric(), dt=dt, maxsubsteps=200, equilibrium_tolerance=tolerance)
    for _ in range(relax):
        mb.step(dt)
    return mb


def test_a_settled_cell_finishes_in_one_relaxed_move(backend):
    # the source relaxed at the species' destruction frequencies is within 1e-6 of the finite-rate source (measured 4e-8) and not it
    explicit = dQ(reactants(backend, "coupled", 0.0))[5:]
    withexit = reactants(backend, "sub-stepped", 1e-7)
    assert int(at_cell(withexit, "substeps")) == 1
    source = dQ(withexit)[5:]
    scale = np.abs(explicit).max()
    assert np.abs(source - explicit).max() < 1e-6 * scale
    assert not np.array_equal(source, explicit)
    # without the exit the one substep covers the step: the finite-rate source to the bit
    assert np.array_equal(dQ(reactants(backend, "sub-stepped", 0.0))[5:], explicit)


def test_a_right_hand_side_before_any_step_is_the_finite_rate_source(backend):
    # the step scalar is nothing until a step sets it, so the exit is not judged over no time
    explicit = dQ(reactants(backend, "coupled", 0.0))[5:]
    bare = cell(backend, mechanism, "sub-stepped", 10e5, 400.0, stoichiometric(), stepset=False, equilibrium_tolerance=1e-7)
    assert np.array_equal(dQ(bare)[5:], explicit)


def test_an_unsettled_cell_substeps_as_the_one_pass_form(backend):
    withexit, without = products(backend, 1e-7), products(backend, 0.0)
    withexit.step(1e-6)
    without.step(1e-6)
    assert np.array_equal(interior(withexit, "substeps"), interior(without, "substeps"))
    assert interior(withexit, "substeps").min() > 1
    assert np.array_equal(at_cell(withexit, "Q"), at_cell(without, "Q"))


def test_the_tolerance_is_a_mass_fraction(backend):
    for bad in (-1e-7, 1.0, "no"):
        with pytest.raises(ConfigError):
            products(backend, bad, relax=0)
