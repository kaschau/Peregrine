import sys
from pathlib import Path

import numpy as np

# this tree's case names to develop's (commit 01384392, the last with src/peregrinepy) and the config sections overridden on its build
names = {
    "navstokes rk3": ("navierStokes rk3", {}),
    "euler rk3": ("euler rk3", {}),
    "euler rk3 ausmup": ("euler rk3", {"RHS": {"primaryAdvFlux": "ausmPlusUp"}}),
    "navstokes rk3 CH4_O2": ("navierStokes rk3 CH4_O2", {}),
    "navstokes rk3 CH4_O2 binary": ("navierStokes rk3 CH4_O2 binary", {}),
    "navstokes rk3 CH4_O2 explicit": ("navierStokes rk3 CH4_O2 explicit", {}),
    # develop's one-pass substepper without its entropy cap, which this tree does not carry
    "navstokes rk3 CH4_O2 sub-stepped": (
        "navierStokes rk3 CH4_O2 explicit",
        {"chemistry": {"source": "substepped", "entropyBisections": 0, "bulkSubSteps": 0, "maxSubSteps": 200, "equilibriumTolerance": 1e-7}},
    ),
}

# cases whose right-hand side is this tree's own, with the measured distance from develop's
unheld = {
    "euler rk3 ausmup": "develop's AUSM+-up forms each side's fourth and fifth degree splits with the other side's second-degree split",
    "euler rk3 muscl-koren-ausmup": "MUSCL-Koren differs from develop by 13 % of the row on the gate's state, roundoff on a smooth field",
}


# develop's build pins the flux, so the config is overridden on its way into the solver
def build(cases, case, sections):
    real = cases.pg.multiBlock.solver

    def solver(config, mesh):
        for section, keys in sections.items():
            for key, value in keys.items():
                config[section][key] = value
        return real(config, mesh)

    cases.pg.multiBlock.solver = solver
    try:
        return cases.build(**cases.cases[case])
    finally:
        cases.pg.multiBlock.solver = real


def record(directory, develop):
    # develop's extraction first on the path, so its tests package and not this tree's is what imports
    sys.path.insert(0, str(develop))
    # peregrinepy exists only in develop's venv: run this under direnv exec ~/PEREGRINE/develop, on a git archive of 01384392
    from peregrinepy.backend import abi
    from tests.gate import cases

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    abi.lib.initialize()
    for ours, (theirs, sections) in names.items():
        mb = build(cases, theirs, sections)
        arrays = {}
        for blk in mb.blocks:
            arrays[f"Q{blk.nblki}"] = blk.Q.get()
            arrays[f"started{blk.nblki}"] = blk.started
        # a bare right-hand side sizes its substeps by the step scalar, which a step sets
        mb.integrator.dt.set(mb.config["simulation"]["dt"])
        mb.rhs()
        ng = mb.ng
        for blk in mb.blocks:
            arrays[f"dQ{blk.nblki}"] = blk.dQ.get()[ng:-ng, ng:-ng, ng:-ng].copy()
        np.savez(directory / f"{ours}.npz", **arrays)
        print(f"{ours}: {theirs} {sections}, {len(mb.blocks)} blocks of {mb.sim.ne} rows", flush=True)
    abi.lib.finalize()


if __name__ == "__main__":
    develop = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd()
    record(sys.argv[1], develop)
