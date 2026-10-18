from pathlib import Path

import h5py
import yaml

from peregrine.config import Config
from peregrine.misc import Progress


class RestartReader:
    def __init__(self, fname, quiet=True):
        self.fname = fname
        self.quiet = quiet
        with h5py.File(self.fname, "r") as f:
            self.nrt, self.tme = int(f.attrs["nrt"]), float(f.attrs["tme"])

            def names(key):
                return [s.decode() for s in f.attrs[key]]

            # the least a case starts from, and everything the file holds
            self.privars, self.variables = names("privars"), names("variables")
            # the grid it sits on, and the case that wrote it, when it was one
            self.grid = str(Path(fname).parent / f.attrs["grid"])
            self.text = f.attrs["config"]

    # parsed only for a run that starts from the file without a config
    @property
    def cfg(self):
        return Config.from_dict(yaml.safe_load(self.text)) if self.text else None

    def fill(self, mb, iovars):
        bar = Progress(len(mb.blocks), self.quiet)
        with h5py.File(self.fname, "r") as f, bar:
            for blk in mb.blocks:
                results = f[f"results_{blk.origblk:06d}"]
                dest = blk.q.get()
                slab = blk.cellslab
                for i, (var, default) in enumerate(iovars.primitives.items()):
                    if var not in results:
                        if default is None:
                            raise ValueError(f"{self.fname} holds no {var}, which a case starts from")
                        dest[blk.interior + (i,)] = default
                    # a device-layout array reads straight in; a host-layout one copies through a temporary
                    elif dest.flags["F_CONTIGUOUS"]:
                        results[var].read_direct(
                            dest.T,
                            source_sel=slab,
                            dest_sel=(i,) + blk.interior,
                        )
                    else:
                        dest[blk.interior + (i,)] = results[var][slab].T

                blk.q.set(dest)
                bar.step(f"Reading in block {blk.nblki}")
