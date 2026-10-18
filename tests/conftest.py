import peregrine as pg

import pytest


def pytest_addoption(parser):
    names = sorted(c.name for c in pg.misc.subclasses(pg.backend.BaseBackend) if c.name)
    parser.addoption("--backend", choices=names, help="the backend the tests run on")


@pytest.fixture(scope="session")
def backend(request):
    name = request.config.getoption("backend")
    if name is None:
        message = "--backend names the backend the tests run on, e.g. --backend openmp"
        pytest.exit(message, returncode=pytest.ExitCode.USAGE_ERROR)
    made = []

    def make(config):
        made.append(pg.get_backend(name, config))
        return made[-1]

    # the name, for a test that spawns a script under mpiexec
    make.name = name
    yield make
    for b in made:
        b.stop()
