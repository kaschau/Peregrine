import numpy as np
import pytest

from tests.eos.support import cubicroots, cubic_ab, setstate, solver, state, textbookdensity

species = ["O2", "N2", "CO2", "CH4", "H2"]


@pytest.fixture(scope="module")
def mb(backend):
    return solver(backend, species, "real-gas")


@pytest.mark.parametrize("p", [1e5, 1e7, 6e7])
@pytest.mark.parametrize("T", [400.0, 1000.0, 3000.0])
def test_density_is_the_textbook_cubics(mb, p, T):
    sp = [mb.sim.mixture.species[n] for n in species]
    Y = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
    A, B, _ = cubic_ab(sp, p, T, Y)
    roots = cubicroots(A, B)
    # at 1000 K and beyond two more real roots sit below the covolume, which the eos must leave alone
    assert np.sum(roots > B) == 1
    setstate(mb, p, T, Y)
    assert state(mb)["rho"] == pytest.approx(textbookdensity(sp, p, T, Y), rel=1e-10)
