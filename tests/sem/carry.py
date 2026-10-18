import json
import sys
from pathlib import Path

import numpy as np

from tests.sem.gate import CASES, REFERENCES

# the previous implementation's option names, and this tree's
renamed = {"populationMethod": "population", "interpolate": "interp", "cEddy": "ceddy"}


def translate(params):
    old = json.loads(str(params))
    new = {renamed.get(k, k): v for k, v in old.items()}
    if "population" in new:
        new["population"] = new["population"].lower()
    return json.dumps(new)


def gate_params(case):
    _, _, _, tme, kwargs = CASES[case]()
    return json.dumps(dict(totalTime=tme, **kwargs))


def carry(source):
    REFERENCES.mkdir(exist_ok=True)
    for path in sorted(Path(source).glob("*.npz")):
        ref = dict(np.load(path))
        if "params" in ref:
            ref["params"] = np.array(translate(ref["params"]))
            assert str(ref["params"]) == gate_params(path.stem), path.stem
        np.savez(REFERENCES / path.name, **ref)
        print(f"{path.name}: {ref['params'] if 'params' in ref else 'no params'}")


if __name__ == "__main__":
    carry(sys.argv[1])
