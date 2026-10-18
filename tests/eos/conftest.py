import pytest

from tests.eos.support import solver


# one solver for every dense module: each loads kernels into this process, and macOS caps the thread-local keys they take
@pytest.fixture(scope="session")
def dense(backend):
    return solver(backend, ["O2", "CH4", "N2", "CO2"], "real-gas", (100.0, 1000.0))
