import sys
from functools import partial
from pathlib import Path

import peregrine as pg
from mpi4py import MPI
from peregrine.partition import get_partitioner

from tests.gate import cases
from tests.gate.test_gate import load


def make(directory, *counts):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mb = pg.multiblock.Grid()
    cases.mesh().fill(mb)
    writer = pg.writers.GridWriter(mb, str(directory / "g.h5"), precision="double")
    writer.write(mb)
    topology = pg.multiblock.BaseMultiBlock.from_grid(f"{directory}/g.h5")
    for ranks in counts:
        groups = get_partitioner().partition(topology, ranks, ranks)
        writer.add_partition(topology, groups, ranks)


def run(directory, name, backend):
    comm = MPI.COMM_WORLD
    mb = cases.build(
        partial(pg.get_backend, backend),
        **cases.cases[name],
        ranks=(comm.size, comm.size),
        grid=f"{directory}/g.h5",
    )
    cases.step(mb)
    mine = cases.block_digests(mb)
    every = comm.gather(mine, root=0)
    if comm.rank == 0:
        got = cases.digest({k: v for d in every for k, v in d.items()})
        platform = cases.platform(mb)
        want = load().get(platform, {}).get(name)
        verdict = "matches" if got == want else f"differs from {want}"
        print(f"{name} on {comm.size} ranks ({platform}): digest {got} {verdict}", flush=True)
    mb.backend.stop()


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2], *map(int, sys.argv[3:]))
    else:
        run(sys.argv[2], sys.argv[3], sys.argv[4])
