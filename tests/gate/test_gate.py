import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.gate import cases
from tests.gate.develop import unheld

# a moved hash is evidence, not a verdict: decide on the merits, then re-record
references = Path(__file__).with_name("references.yaml")
develop = os.environ.get("PG_GATE_DEVELOP")


def load():
    return yaml.safe_load(references.read_text()) if references.exists() else {}


def names(held=False):
    params = []
    for name, case in cases.cases.items():
        if held and name in unheld:
            params.append(pytest.param(name, marks=pytest.mark.skip(reason=unheld[name])))
        else:
            params.append(name)
    return params


@pytest.mark.parametrize("name", names())
def test_gate(backend, name):
    mb = cases.build(backend, **cases.cases[name])
    cases.step(mb)
    got = cases.digest(cases.block_digests(mb))
    platform = cases.platform(mb)
    known = load()
    if os.environ.get("PG_GATE_RECORD"):
        known.setdefault(platform, {})[name] = got
        references.write_text(yaml.safe_dump(known, sort_keys=True))
        return
    if platform not in known or name not in known[platform]:
        pytest.skip(
            f"no reference for {name!r} on {platform}; record with PG_GATE_RECORD=1"
        )
    assert (
        got == known[platform][name]
    ), f"{name} on {platform}: {got} != {known[platform][name]}"


@pytest.mark.skipif(not develop, reason="PG_GATE_DEVELOP names no recording")
@pytest.mark.parametrize("name", names(held=True))
def test_develop(backend, name):
    path = Path(develop) / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"no {path}; record it with tests/gate/develop.py")
    ref = np.load(path)
    mb = cases.build(backend, **cases.cases[name])
    tpg = mb.sim.mixture.eos.name == "tpg"
    viscous = mb.sim.name == "navier-stokes"
    got = cases.rhs(mb)
    # tpg: the trees' inversions of the start state differ by 1.6e-13 in energy, which the pressure carries into the momentum rows at 9e-11; KEPaEC's thermal-energy average is this tree's own, so the energy row is not held
    tol = np.full(mb.sim.ne, 1e-13)
    if tpg:
        tol[1:4] = 1e-9
        tol[4] = np.inf
    # the reactions' sensitivity to T carries that inversion difference into the species rows at 8.5e-12
    if "chemint" in cases.cases[name]:
        tol[5:] = 1e-10
    # develop's diffusive energy flux carries the stress work with the wrong sign
    if viscous:
        tol[4] = np.inf
    for blk in mb.blocks:
        # the wall cells take this tree's own wall flux (acoustic wall pressure, two-point viscous rows), so they are not held to develop's
        held = np.ones(got[blk.nblki].shape[:3], dtype=bool)
        for intface in blk.intfaces:
            if intface.nbr is None:
                plane = [slice(None)] * 3
                plane[intface.axis] = 0 if intface.is_low else -1
                held[tuple(plane)] = False
        Q = blk.Q.get()[blk.interior][held]
        Qdev = ref[f"Q{blk.nblki}"][blk.interior][held]
        dQ, want = got[blk.nblki][held], ref[f"dQ{blk.nblki}"][held]
        # the energies differ by a reference per unit mass (cpg's h298), which the continuity row carries into the energy row
        h0 = (Q[..., 4] - Qdev[..., 4]) / Q[..., 0]
        dQ[..., 4] -= h0 * dQ[..., 0]
        scale = np.abs(want).reshape(-1, mb.sim.ne).max(axis=0)
        off = np.abs(dQ - want).reshape(-1, mb.sim.ne).max(axis=0) / scale
        print(f"{name} block {blk.nblki}: " + " ".join(f"{o:.1e}" for o in off))
        bad = np.where(off > tol)[0]
        assert not bad.size, f"{name} block {blk.nblki}: rows {bad.tolist()} off by {off[bad]}"
