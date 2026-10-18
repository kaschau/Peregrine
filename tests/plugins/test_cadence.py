from types import SimpleNamespace

import numpy as np

from peregrine.plugins import Cadence


def held(value):
    return SimpleNamespace(get=lambda: value)


def at(nrt, tme, dt):
    return SimpleNamespace(nrt=nrt, tme=held(tme), dt=held(dt))


def test_by_steps():
    c = Cadence({"niter-out": 3})
    assert [c.due(at(n, 0.0, 0.0)) for n in range(1, 7)] == [0, 0, 1, 0, 0, 1]
    assert c.next(at(1, 0.0, 0.0)) is None
    assert Cadence({}).niterout == 1


def test_by_time():
    c = Cadence({"dt-out": 1e-4})
    # a step that crosses a multiple is due, one that does not is not
    assert c.due(at(1, 1.3e-4, 0.5e-4)) and not c.due(at(2, 1.8e-4, 0.5e-4))
    assert c.next(at(2, 1.8e-4, 0.5e-4)) == 2e-4


def test_a_landing_within_rounding_is_on_the_multiple():
    c = Cadence({"dt-out": 1e-4})
    short = 3e-4 - 2 * np.spacing(3e-4)
    assert c.nreached(short) == 3 and c.due(at(3, short, 0.7e-4))
    # and the step after it does not fire again
    assert not c.due(at(4, short + 0.2e-4, 0.2e-4))
    assert c.next(at(3, short, 0.7e-4)) == 4e-4
