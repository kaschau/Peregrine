import numpy as np
import pytest

from tests.chemint import cases


# each kind at a step it survives explicitly: the sub-stepped one is stiff at the duct's own
@pytest.mark.parametrize("kind, dt", [("coupled", 1e-9), ("sub-stepped", 1e-6)])
def test_a_claim_on_the_derivative_is_spent_by_another_registers_rhs(backend, kind, dt):
    mb = cases.duct(backend, kind, dt=dt)
    mb.step(dt)
    Q, Q1, dQ = mb.regs[0], mb.regs[1], mb.regs[-1]
    mb.rhs(Q=Q, dQ=dQ)
    clean = cases.interior(mb, dQ).copy()
    mb.prerhs(Q, dQ)
    mb.rhs(Q=Q1, dQ=dQ)
    mb.rhs(Q=Q, dQ=dQ)
    # walls only, so every halo is rebuilt from the interior at each readying and the right-hand side is the clean one to the bit
    assert np.isfinite(clean).all()
    assert np.array_equal(cases.interior(mb, dQ), clean)
