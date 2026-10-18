import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import peregrine as pg

from tests.communication.test_orientation import ReorientedPair

STEPPER = Path(__file__).parent / "stepranks.py"

# the reoriented pairs make the landing re-index out of the buffer
cases = [
    ("euler", "constant", "123"),
    ("euler", "muscl", "123"),
    ("euler", "muscl", "162"),
    ("euler", "constant", "453"),
    ("navier-stokes", "constant", "123"),
    ("navier-stokes", "muscl", "123"),
]


# the timeout makes a hung job fail the test rather than the run
def stepper(backend, grid, ranks, out, simulator, recon):
    cmd = ["mpiexec", "-n", str(ranks), "--oversubscribe", sys.executable, str(STEPPER), grid, str(ranks), out, simulator, recon, backend.name]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="no mpiexec")
@pytest.mark.parametrize("simulator,recon,S", cases, ids=["-".join(c) for c in cases])
def test_two_ranks_step_as_one(backend, tmp_path, simulator, recon, S):
    mesher = ReorientedPair(S, mbdims=[2, 1, 1], blkdims=[9, 6, 5], lengths=[2, 1, 1], defaultbc="wall")
    mb = pg.multiblock.Grid()
    mesher.fill(mb)
    grid = str(tmp_path / "step.h5")
    writer = pg.writers.GridWriter(mb, grid, precision="double")
    writer.write(mb)
    topology = pg.multiblock.BaseMultiBlock.from_grid(grid)
    writer.add_partition(topology, [[0, 1]], 1)
    writer.add_partition(topology, [[0], [1]], 2)
    outs = {}
    for ranks in (1, 2):
        outs[ranks] = str(tmp_path / f"step-{ranks}.npz")
        stepper(backend, grid, ranks, outs[ranks], simulator, recon)
    one, two = np.load(outs[1]), np.load(outs[2])
    assert sorted(one.files) == sorted(two.files) == ["0", "1"]
    for n in one.files:
        assert np.isfinite(two[n]).all()
        assert np.array_equal(one[n], two[n])
