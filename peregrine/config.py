import yaml

from peregrine.misc import FrozenDict, get_comm_rank_size


class ConfigError(Exception):
    def __init__(self, setting, msg):
        text = f"Invalid PEREGRINE setting {setting}: {msg}"
        super().__init__(f"\n\n*****\n{text}\n*****\n\n")


class Config(FrozenDict):
    def __init__(self):
        self["simulation"] = FrozenDict(
            {
                "simulator": "navier-stokes",
                "precision": "double",
                "niter": 1,
                "controller": "fixed",
                "dt": 1e-3,
                "dt-max": 1e-3,
                "cfl": 0.1,
            }
        )
        self["mixture"] = FrozenDict(
            {
                "species": None,
                "eos": "cpg",
                "trans": None,
                "diffusion": "lewis",
                "mixing-rule": "wilke",
                "chemistry": None,
                "bath": None,
                "T-range": None,
                "refit-tol": 1e-3,
                "refit-max-deg": 6,
            }
        )
        self["chem-integration"] = FrozenDict(
            {
                "kind": None,
                "max-substeps": 200,
                "bulk-substeps": 1,
                "stiff-bins": 8,
                "equilibrium-tolerance": 1e-7,
            }
        )
        self["ics"] = FrozenDict(
            {"p": 101325.0, "u": 0.0, "v": 0.0, "w": 0.0, "T": 300.0, "Y": {}}
        )
        self["time-integration"] = FrozenDict({"integrator": "tvd-rk3"})
        self["adv-flux"] = FrozenDict(
            {
                "formulation": "single",
                "primary": "kepaec",
                "recon": "constant",
                "limiter": None,
                "secondary": None,
                "correction": None,
                "switch": None,
                "values": {},
            }
        )
        self["diff-flux"] = FrozenDict(
            {
                "formula": "alpha-damping",
                "values": {},
            }
        )
        # a run reads the section of the backend it is given
        self["backend-openmp"] = FrozenDict({"cxx": "c++", "tile-cells": 128, "tile-elements": 1024, "constform": "literal"})
        self["backend-cuda"] = FrozenDict(
            {
                "tile-cells": 128,
                "tile-elements": 1024,
                "launch-threads": 256,
                "launch-waves": 1,
                "constform": "literal",
            }
        )
        self["backend-hip"] = FrozenDict(
            {
                "tile-cells": 128,
                "tile-elements": 1024,
                "launch-threads": 256,
                "launch-waves": 2,
                "constform": "literal",
            }
        )
        self["halo-exchange"] = FrozenDict({"kind": "hoststaged"})
        # the names under these two are the case's own, so they are not frozen
        self["bcvals"] = {}
        self["plugins"] = {}

        for key in self.keys():
            if isinstance(self[key], FrozenDict):
                self[key]._freeze()
        self._freeze()

    @classmethod
    def from_dict(cls, given):
        config = cls()
        for section, entries in (given or {}).items():
            for key, value in (entries or {}).items():
                config[section][key] = value
        config.validate()
        return config

    def to_dict(self):
        return {
            section: dict(entries) if isinstance(entries, dict) else entries
            for section, entries in self.items()
        }

    def validate(self):
        self["simulation"]["dt"] = float(self["simulation"]["dt"])


def read_cfg(fname="./peregrine.yaml"):
    comm, rank, size = get_comm_rank_size()
    text = open(fname).read() if rank == 0 else None
    return Config.from_dict(yaml.safe_load(comm.bcast(text, root=0)))


class ConfigDumper(yaml.SafeDumper):
    # a blank line between top-level sections, after https://stackoverflow.com/a/44284819
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()


def write_cfg(cfg, fname):
    with open(fname, "w") as f:
        yaml.dump(cfg.to_dict(), f, Dumper=ConfigDumper, sort_keys=False)
