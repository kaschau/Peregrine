import numpy as np

from peregrine.plugins.solver.base import BaseSolverPlugin


class Catalyst(BaseSolverPlugin):
    name = "catalyst"

    def start(self, mb):
        cfgsect = self.cfgsect
        # only a case that asks for catalyst needs paraview importable
        from paraview import vtk
        from paraview.catalyst import bridge
        from paraview.modules import vtkPVCatalyst as catalyst
        from paraview.vtk.util import numpy_support

        self.vtk, self.bridge, self.catalyst = vtk, bridge, catalyst
        self.numpy_support = numpy_support
        bridge.initialize()
        bridge.add_pipeline(cfgsect["script"])

        self._coproc = bridge.coprocessor

        self.desc = self.catalyst.vtkCPDataDescription()
        self.desc.AddInput("input")

        mbds = self.vtk.vtkMultiBlockDataSet()
        mbds.SetNumberOfBlocks(mb.nblocks)
        for i in range(mb.nblocks):
            mbds.SetBlock(i, None)

        for blk in mb.blocks:
            grid = self.vtk.vtkStructuredGrid()
            grid.SetDimensions(blk.ni, blk.nj, blk.nk)
            interior = blk.nodes.get()[blk.interior]
            xyz = [interior[..., n].ravel(order="F") for n in range(3)]
            coords = np.column_stack(xyz)
            points = self.vtk.vtkPoints()
            points.SetData(self.numpy_support.numpy_to_vtk(coords))
            grid.SetPoints(points)
            for name, array in self._arrays(mb, blk).items():
                self._add(grid, name, array)
            mbds.SetBlock(blk.nblki, grid)

        self.desc.GetInputDescriptionByName("input").SetGrid(mbds)

    def _arrays(self, mb, blk):
        def flat(a):
            return a[blk.interior].ravel(order="F")

        iovars = mb.sim.iovars
        names = iovars.exported
        values = iovars.data(blk, names)
        arrays = {n: flat(values[n]) for n in iovars.scalars_among(names)}
        for name, components in iovars.vectors_among(names).items():
            cols = [flat(values[n]) for n in components]
            arrays[name] = np.column_stack(cols)
        return arrays

    def _add(self, grid, name, array):
        arr = self.numpy_support.numpy_to_vtk(array)
        arr.SetName(name)
        grid.GetCellData().AddArray(arr)

    def _swap(self, grid, name, array):
        grid.GetCellData().RemoveArray(name)
        self._add(grid, name, array)

    def __call__(self, mb):
        self.desc.SetTimeData(mb.tme.get(), mb.nrt)

        if not self._coproc.RequestDataDescription(self.desc):
            return

        mbds = self.desc.GetInputDescriptionByName("input").GetGrid()
        for blk in mb.blocks:
            grid = mbds.GetBlock(blk.nblki)
            for name, array in self._arrays(mb, blk).items():
                self._swap(grid, name, array)

        self._coproc.CoProcess(self.desc)

    def finalize(self, mb):
        self.bridge.finalize()
