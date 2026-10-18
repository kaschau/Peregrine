import cantera as ct
import numpy as np
import pytest

from tests.dense import support

species = ["O2", "N2", "CH4", "H2"]


def diffusivities(mb, p, T, X):
    _, row = support.transport(mb, p, T, support.massfrac(support.species(mb), X))
    return row[2:]


# few solvers: each loads its kernels into this process, and macOS caps the thread-local keys they take
@pytest.fixture(scope="module")
def takahashi(backend):
    return support.solver(backend, species, diffusion="takahashi")


@pytest.fixture(scope="module")
def binary(backend):
    return support.solver(backend, species, diffusion="binary")


# a real gas phase gives Cantera the critical points its correction reads; its Peng-Robinson density does not enter the coefficients
@pytest.fixture(scope="module")
def cantera():
    phase = """
    phases:
    - name: gas
      thermo: Peng-Robinson
      species: [{gri30.yaml/species: [O2, N2, CH4, H2]}]
      state: {T: 300.0, P: 101325.0}
    """
    return ct.Solution(yaml=phase, transport_model="high-pressure")


X = np.array([0.6, 0.2, 0.1, 0.1])


def test_at_low_pressure_it_is_the_binary_model(takahashi, binary):
    # below a reduced pressure of a tenth the correction is one, exactly
    D = diffusivities(takahashi, support.atm, 300.0, X)
    assert np.array_equal(D, diffusivities(binary, support.atm, 300.0, X))


# the binary coefficients underneath are refits of Cantera's, hence the tolerance
@pytest.mark.parametrize("p, T", [(60e6, 300.0), (60e6, 800.0), (20e6, 400.0)])
def test_at_high_pressure_it_is_canteras_correction(takahashi, cantera, p, T):
    cantera.TPX = T, p, dict(zip(species, X))
    D = diffusivities(takahashi, p, T, X)
    assert D == pytest.approx(cantera.mix_diff_coeffs_mass, rel=1.5e-2), (p, T)
