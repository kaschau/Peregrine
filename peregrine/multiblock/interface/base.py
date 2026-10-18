import numpy as np

from peregrine.misc import get_comm_rank_size


class BaseInterface:
    def __init__(self, nface):
        assert 1 <= nface <= 6, "nface must be between (1,6)"

        self.nface = nface
        # the name a grid gives a boundary, for the case to say what it is by
        self.bcname = None
        self.rot = None
        self.nbr = None
        # which rank holds the neighbor
        self.nbrrank = None
        self.orient = None
        self.xlate = None

    @property
    def intkind(self):
        if self.nbr is None:
            return None
        if self.rot is None:
            return "interior"
        return "periodic-rot" if self.rotperiodic else "periodic-xlate"

    ###########################################################################
    # How a halo arriving through this interface is moved onto it
    ###########################################################################
    def set_periodic(self, rot=None, xlate=None):
        if xlate is None:
            xlate = np.zeros(3)
        if rot is None:
            rot = np.eye(3)
        self.xlate = np.array(xlate, np.float64)
        self.rot = np.array(rot, np.float64)

    @property
    def nbrplane(self):
        theiraxis = (self.nbrnface - 1) // 2
        codes = list(self.nbrorient)
        del codes[theiraxis]
        return [self.signedaxis(code) for code in codes]

    @property
    def planeaxes(self):
        return tuple(m for m in range(3) if m != self.axis)

    @property
    def nbr_plane_align(self):
        theirs = self.nbrplane
        transposed = theirs[0][0] == self.planeaxes[1]
        flipped = tuple(m for m, (_, flip) in enumerate(theirs) if flip)
        return transposed, flipped

    # the inverse of laying ours out as they read it, so the flips come off before the transpose
    def align(self, plane):
        transposed, flipped = self.nbr_plane_align
        if flipped:
            plane = np.flip(plane, flipped)
        if transposed:
            plane = np.moveaxis(plane, (0, 1), (1, 0))
        return plane

    def plane(self, index):
        return (slice(None),) * self.axis + (index,)

    @property
    def firstplane(self):
        return self.plane(0 if self.is_low else -1)

    # not nbr: which block it meets is the caller's to set
    def copy_from(self, other):
        self.bcname = other.bcname
        self.orient = other.orient
        if other.rot is None:
            self.rot = None
            self.xlate = None
        else:
            self.set_periodic(rot=other.rot, xlate=other.xlate)

    def set_interior(self, nbr, orient="123"):
        self.bcname = None
        self.nbr = nbr
        self.orient = orient
        self.rot = None
        self.xlate = None

    # Rodrigues' formula: http://paulbourke.net/geometry/rotate/
    @staticmethod
    def rotabout(axis, deg):
        u = np.array(axis, np.float64)
        u = u / np.linalg.norm(u)
        th = deg * np.pi / 180.0
        ct, st = np.cos(th), np.sin(th)
        x, y, z = u
        cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        return ct * np.eye(3) + st * cross + (1.0 - ct) * np.outer(u, u)

    @staticmethod
    def signedaxis(code):
        n = int(code)
        return (n - 1) % 3, n > 3

    @staticmethod
    def orientcode(axis, flip):
        return str(axis + 1 + (3 if flip else 0))

    @property
    def axis(self):
        return (self.nface - 1) // 2

    @property
    def is_low(self):
        return self.nface % 2 == 1

    @property
    def rotperiodic(self):
        return self.rot is not None and not np.allclose(self.rot, np.eye(3))

    @property
    def offrank(self):
        return self.nbr is not None and self.nbrrank != get_comm_rank_size()[1]

    @property
    def nbrnface(self):
        axis, flip = self.signedaxis(self.orient[self.axis])
        # an aligned neighbor meets our low interface with its high one, a flipped one with its low
        return 2 * axis + (1 if self.is_low == flip else 2)

    @property
    def nbrorient(self):
        out = [None, None, None]
        for ours, code in enumerate(self.orient):
            theirs, flip = self.signedaxis(code)
            out[theirs] = self.orientcode(ours, flip)
        assert None not in out, "orientation does not name three distinct axes"
        return "".join(out)

    @property
    def align_nbr_by(self):
        perm, flips = [], []
        for code in self.orient:
            axis, flip = self.signedaxis(code)
            perm.append(axis)
            flips.append(flip)
        return perm, flips
