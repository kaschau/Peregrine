import numpy as np

from peregrine.multiblock.interface.base import BaseInterface


class GridInterface(BaseInterface):
    def __init__(self, nface, blk):
        super().__init__(nface)
        self.blk = blk

    @property
    def ng(self):
        return self.blk.ng

    ###########################################################################
    # The planes of a block array at this interface, layer 0 nearest it
    ###########################################################################
    def halo(self, array):
        ng = self.ng
        planes = np.moveaxis(array, self.axis, 0)
        # counted from the end explicitly: a negative slice of zero, [-0:], is the whole array
        return planes[:ng][::-1] if self.is_low else planes[len(planes) - ng :]

    def faceplane(self, array):
        ng = self.ng
        planes = np.moveaxis(array, self.axis, 0)
        plane = planes[ng if self.is_low else len(planes) - ng - 1]
        a, b = plane.shape[:2]
        return plane[ng : a - ng, ng : b - ng]
