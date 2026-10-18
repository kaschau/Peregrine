import argparse

import yaml

import peregrine as pg
from peregrine.tools.channel import ChannelTool, template

from tests.gases import configure


def test_the_channel_names_its_walls_and_repeats_along_x_and_z(backend, tmp_path):
    spec = yaml.safe_load(template)
    # a small channel: coarse streamwise and spanwise wall units, and a friction Reynolds number just past the log layer's reach
    spec.update({"delta": 0.1, "retau": 300.0, "lx": 6, "z-dim": 3, "dx": 100, "dz": 100, "dy-core": 20})
    (tmp_path / "channel.yaml").write_text(yaml.safe_dump(spec))
    grid = str(tmp_path / "g.h5")
    ChannelTool().main(argparse.Namespace(spec=str(tmp_path / "channel.yaml"), out=grid, template=None))
    config = pg.Config()
    configure(config, "air", "euler")
    config["bcvals"]["wall"] = {"kind": "slp-adia-wall"}
    mb = pg.get_solver(config, pg.readers.GridReader(grid), backend(config))
    (blk,) = mb.blocks
    names = {f.nface: f.bcname for f in blk.intfaces if f.nbr is None}
    assert names == {3: "wall", 4: "wall"}
    assert all(f.nbr == blk.nblki for f in blk.intfaces if f.nface in (1, 2, 5, 6))
    mb.step(1e-7)
