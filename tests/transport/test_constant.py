import numpy as np
import pytest

from tests.gases import air
from tests.transport import cases

two = {
    "H2": {"MW": 2.016, "cp0": 14300.0, "mu0": 8.9e-6, "kappa0": 0.18},
    "N2": {"MW": 28.014, "cp0": 1040.0, "mu0": 1.78e-5, "kappa0": 0.026},
}


def test_one_species(backend):
    mb = cases.solver(backend, air, eos="cpg", trans="constant")
    assert mb.sim.mixture.trans.name == "constant"
    p, T, Y = cases.draw(8, 1)
    mu, kappa = cases.set_state(mb, p, T, Y)[:2]
    assert mu == air["Air"]["mu0"]
    assert kappa == air["Air"]["kappa0"]


@pytest.mark.parametrize("YH2", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("rule", ["wilke", "herning"])
def test_two_species(backend, rule, YH2):
    mb = cases.solver(backend, two, eos="cpg", trans="constant", **{"mixing-rule": rule})
    assert mb.sim.mixture.mixrule.name == rule
    p, T, _ = cases.draw(9, 2)
    Y = np.array([YH2, 1.0 - YH2])
    mu, kappa = cases.set_state(mb, p, T, Y)[:2]
    c = mb.sim.mixture.constants()
    X = cases.molefrac(Y, c["MW"])
    # the code gives 2e-16; a transposed pair constant moves mu by percents
    assert mu == pytest.approx(cases.rules[rule](c["mu0"], X, c["MW"]), rel=1e-12)
    assert kappa == pytest.approx(cases.kappamix(c["kappa0"], X), rel=1e-12)
