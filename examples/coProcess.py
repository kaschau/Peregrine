#!/usr/bin/env python
"""

run with 2 mpi processes!!!!!!

    mpiexec -n 2 python examples/coProcess.py <backend>

"""

from mpi4py import MPI  # noqa: F401

import sys

import peregrine as pg
import os

fname = """
# script-version: 2.0
# Catalyst state generated using paraview version 5.10.0

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XDMF Reader'
# qxmf = XDMFReader(registrationName="input")
qxmf = TrivialProducer(registrationName="input")
qxmf.CellArrayStatus = ["Air", "T", "Velocity", "p", "rho"]

# create a new 'Slice'
slice1 = Slice(registrationName="Slice1", Input=qxmf)
slice1.SliceType = "Plane"
slice1.HyperTreeGridSlicer = "Plane"
slice1.Triangulatetheslice = 0
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [0.1, 0.01, 0.0005]
slice1.SliceType.Normal = [0.0, 0.0, 1.0]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [0.1, 0.01, 0.0005]

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor
vTM1 = CreateExtractor("VTM", slice1, registrationName="VTM1")
# trace defaults for the extractor.
vTM1.Trigger = "TimeStep"

# init the 'TimeStep' selected for 'Trigger'
vTM1.Trigger.Frequency = 10

# init the 'VTM' selected for 'Writer'
vTM1.Writer.FileName = "Slice1_{timestep:06d}.vtm"

# ----------------------------------------------------------------
# restore active source
SetActiveSource(vTM1)
# ----------------------------------------------------------------

# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst



options = catalyst.Options()
options.ExtractsOutputDirectory = "./"
options.GlobalTrigger = "TimeStep"
options.CatalystLiveTrigger = "TimeStep"

# init the 'TimeStep' selected for 'GlobalTrigger'
options.GlobalTrigger.Frequency = 10

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    from paraview.simple import SaveExtractsUsingCatalystOptions

    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
"""

# a calorically perfect air, stated in full: the library carries no such species
air = {
    "Air": {
        "MW": 28.97,
        "cp0": 1002.838449439523,
        "mu0": 1.8591191080521142e-05,
        "kappa0": 0.02625394405190068,
    }
}


class RankMesher(pg.mesher.CubeMesher):
    """A row of blocks along i, one per rank: each rank meshes the block of
    its number and joins it to its neighbors across the ranks."""

    def fill(self, mb):
        rank = pg.misc.get_comm_rank_size()[1]
        blk = mb.add_blk(rank)
        mb.nblocks = self.nblks
        self.shape_blk(blk, rank, 0, 0)
        self.connect(blk, rank, (rank, 0, 0))
        for face in blk.intfaces:
            face.nbrrank = face.nbr
        self.set_periodics(blk, rank, 0, 0)


def simulate(name):
    """A channel of two blocks, fed at one end and let out the other, with
    Catalyst watching every step."""
    config = pg.Config()
    config["simulation"]["simulator"] = "navier-stokes"
    config["mixture"]["species"] = air
    config["mixture"]["trans"] = "constant"
    config["plugins"]["catalyst"] = {"script": "tempcoproc.py"}
    config["ics"]["u"] = 10.0
    config["bcvals"]["inlet"] = {
        "kind": "sub-in-fvt",
        "u": 10.0,
        "v": 0.0,
        "w": 0.0,
        "T": 300.0,
    }
    config["bcvals"]["exit"] = {
        "kind": "sub-out-fp",
        "p": 101325.0,
    }
    config["bcvals"]["walls"] = {"kind": "no-slp-adia-wall"}
    config.validate()
    backend = pg.get_backend(name, config)

    comm, rank, size = pg.misc.get_comm_rank_size()
    if rank == 0:
        with open("tempcoproc.py", "w") as f:
            f.write(fname)
    # the plugin reads the pipeline on every rank
    comm.Barrier()
    mesh = RankMesher(
        mbdims=[2, 1, 1],
        blkdims=[100, 40, 2],
        lengths=[0.2, 0.02, 0.001],
        periodic=[False, False, True],
        bcnames={1: "inlet", 2: "exit", 3: "walls", 4: "walls"},
    )
    mb = pg.get_solver(config, mesh, backend)

    if rank == 0:
        print(mb)
    dt = 1.44e-6
    catalyst = mb.plugins["catalyst"]
    catalyst(mb)
    bar = pg.misc.Progress(100, quiet=rank != 0)
    while mb.nrt < 100:
        bar.at(mb.nrt)
        mb.step(dt)
        catalyst(mb)

    catalyst.finalize(mb)
    if rank == 0:
        os.remove("./tempcoproc.py")
    backend.stop()


if __name__ == "__main__":
    try:
        simulate(sys.argv[1])

    except Exception as e:
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
