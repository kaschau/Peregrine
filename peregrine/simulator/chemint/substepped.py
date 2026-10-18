import math

import numpy as np

from peregrine.config import ConfigError
from peregrine.launch.kernel import Kernel
from peregrine.launch.ranges import CellCenterRange
from peregrine.simulator.chemint.base import BaseReactingChemistry, templates


class SubsteppedChemistry(BaseReactingChemistry):
    name = "sub-stepped"

    def __init__(self, cfgsect, mixture):
        super().__init__(cfgsect, mixture)
        keys = ("max-substeps", "bulk-substeps", "stiff-bins", "equilibrium-tolerance")
        cap, bulk, bins, tol = (cfgsect[k] for k in keys)
        # a substep below this share of the step is none
        self.options = dict(cap=cap, bulk=bulk, bins=bins, eqtol=tol, stepfrac=1e-12)

    def validate(self):
        super().validate()
        cap, bulk, bins, tol = (self.options[k] for k in ("cap", "bulk", "bins", "eqtol"))
        if not isinstance(cap, int) or cap < 1:
            raise ConfigError("chem-integration.max-substeps", f"{cap!r} is not a count of one or more")
        if not isinstance(bulk, int) or not 1 <= bulk <= cap:
            raise ConfigError("chem-integration.bulk-substeps", f"{bulk!r} is not a count from one to max-substeps")
        if not isinstance(bins, int) or not 1 <= bins <= 16:
            raise ConfigError("chem-integration.stiff-bins", f"{bins!r} is not a count from one to sixteen")
        if not isinstance(tol, (int, float)) or not 0.0 <= tol < 1.0:
            raise ConfigError("chem-integration.equilibrium-tolerance", f"{tol!r} is not a mass fraction below one, zero for no exit")

    def arrays(self):
        bins = self.options["bins"]
        return {
            # a list read flat: a header of 1 + bins counts, then a segment per bin
            "stiffcells": dict(kind=CellCenterRange, comps=bins, dtype=np.int32),
            # each cell's last count, the next right-hand side's bin key
            "substeps": dict(kind=CellCenterRange, dtype=np.int32),
        }

    @property
    def compacted(self):
        return self.options["bulk"] < self.options["cap"]

    def passkernel(self, name, limit, exit, Q, dQ):
        options = self.options
        cap, bins = options["cap"], options["bins"]
        binscale = bins / math.log(cap / limit) if limit < cap else 0.0
        own = dict(options, limit=limit, exit=exit, limitinv=1.0 / limit, binscale=binscale)
        cells = dict(state="state", stiffcells="stiffcells", substeps="substeps", dt="dt")
        return Kernel(self.provider, f"{templates}.{name}", options=own, Q=Q, dQ=dQ, **cells)

    ###########################################################################
    # Slot graphs
    ###########################################################################
    def begin_dQ_graph(self, g, Q, dQ):
        options = self.options
        after = ()
        if self.compacted:
            reset = Kernel(self.provider, f"{templates}.reset", options=options, stiffcells="stiffcells")
            g.add_launch("reset", reset, "origin", after=after)
            after = ("reset",)
        # the bulk pass lists its stiff cells after the reset, the dense pass takes the list
        g.add_launch("bulk", self.passkernel("bulk", options["bulk"], True, Q, dQ), "interior", after=after)
        if self.compacted:
            # no exit for the dense pass: only a cell's first substep judges the exit, and a listed cell takes that substep in the bulk pass
            g.add_launch("stiff", self.passkernel("stiff", options["cap"], False, Q, dQ), "full", after=("bulk",))
