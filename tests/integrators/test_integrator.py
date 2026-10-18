import pytest

import peregrine as pg

from tests.integrators.cases import box


def test_the_integrator_owns_its_controller(backend):
    config, mesh = box(controller="cfl", cfl=0.5)
    config["simulation"]["dt-max"] = 1e-3
    mb = pg.get_solver(config, mesh, backend(config))
    assert type(mb.integ.controller).name == "cfl"
    # the speeds it sizes by are the physics' kernel, on every case
    assert "cflmax" in mb.kernels
    assert 0 < mb.integ.controller.stepsize(mb) <= 1e-3
    config, mesh = box(controller="fixed", dt=2.5e-7)
    mb = pg.get_solver(config, mesh, backend(config))
    assert type(mb.integ.controller).name == "fixed"
    assert mb.integ.controller.stepsize(mb) == 2.5e-7


@pytest.mark.parametrize(
    "integrator, nregs, nstages",
    [("rk1", 2, 1), ("rk2", 3, 2), ("tvd-rk3", 3, 3), ("rk34", 3, 4), ("rk4", 4, 4)],
)
def test_the_registers_are_declared_on_the_blocks(backend, integrator, nregs, nstages):
    config, mesh = box(integrator=integrator)
    mb = pg.get_solver(config, mesh, backend(config))
    assert mb.integ.nregs == nregs
    assert mb.regs == ("Q",) + tuple(f"Q{n}" for n in range(1, nregs))
    assert mb.regs == tuple(mb.arrays["Q"].banks)
    blk = mb.blocks[0]
    for register in mb.regs:
        assert mb.columns[register] is mb.arrays["Q"]
        assert getattr(blk, register).shape == blk.Q.shape
    stages = mb.integ.stageregs(mb.regs)
    assert len(stages) == nstages
    # a right-hand side per stage's pair, an advance per stage, a postrhs on the solution and on what every stage writes
    assert {k[1:] for k in mb.graphs if k[0] == "rhs"} == {(s.Q, s.dQ) for s in stages}
    assert [k for k in mb.graphs if k[0] == "advance"] == [("advance", mb.regs, n) for n in range(nstages)]
    assert {k[1] for k in mb.graphs if k[0] == "postrhs"} == {"Q", *(s.Qout for s in stages)}
    # the last stage writes the solution; every stage's derivative is the last register
    assert stages[-1].Qout == "Q" and {s.dQ for s in stages} == {mb.regs[-1]}
