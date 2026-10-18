import numpy as np

import peregrine as pg
from peregrine.launch.arrays import ArraySpec
from peregrine.launch.ranges import InterfacePlaneRange
from peregrine.simulator.euler.bcs import BaseEulerBc

from tests.minimal.cases import box


class PlaneTest(BaseEulerBc):
    kind = "plane-test"

    def __init__(self, name, given, sim):
        super().__init__(name, given, sim)
        self.ne = sim.ne

    @property
    def columns(self):
        plane = ArraySpec("plane", InterfacePlaneRange, self.ne)
        return {**super().columns, "plane": plane}

    def start(self, tme, interfaces):
        super().start(tme, interfaces)
        for interface in interfaces:
            interface.plane.set(planeof(interface, self.ne))

    def preadvflux(self, Q):
        return self._kernels("preadvflux", vals="plane", Qhalo=Q)


def planeof(interface, ne):
    a, b = interface.shape
    i, j, row = np.meshgrid(np.arange(a), np.arange(b), np.arange(ne), indexing="ij")
    values = 100.0 * i + 10.0 * j + row
    return values.reshape(interface.plane.shape)


def test_the_halo_takes_the_plane(backend):
    config, mesh = box(dims=(5, 7, 9))
    config["bcvals"]["wall"] = {"kind": "plane-test"}
    mb = pg.get_solver(config, mesh, backend(config))
    blk = mb.blocks[0]
    ng, ne = blk.ng, mb.sim.ne
    mb.apply_bcs("preadvflux")
    Q = blk.Q.get()
    for interface in blk.intfaces:
        want = planeof(interface, ne).reshape(*interface.shape, ne)
        for layer in interface.halo(Q):
            got = layer[ng:-ng, ng:-ng]
            assert np.array_equal(got, want), (interface.nface, got[0, 0], want[0, 0])
