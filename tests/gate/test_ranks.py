import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.gate import ranks

ROOT = Path(__file__).resolve().parents[2]


def digest(directory, name, size, backend):
    cmd = ["mpiexec", "-n", str(size), "--oversubscribe", sys.executable]
    cmd += ["-m", "tests.gate.ranks", "run", str(directory), name, backend]
    # a hung MPI job fails the test rather than the run
    out = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True, timeout=300)
    found = re.search(r"digest (\w+)", out.stdout)
    assert found, out.stdout
    return found.group(1)


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="no mpiexec")
@pytest.mark.parametrize(
    "name", ["euler rk3", "euler rk3 muscl-koren-ausmup", "navstokes rk3 CH4_O2 binary"]
)
def test_two_ranks_hash_as_one(backend, tmp_path, name):
    ranks.make(tmp_path, 1, 2)
    assert digest(tmp_path, name, 2, backend.name) == digest(tmp_path, name, 1, backend.name)
