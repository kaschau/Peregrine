from peregrine.misc import subclass_where
from peregrine.mesher.base import BaseMesher
from peregrine.mesher.cube import CubeMesher
from peregrine.mesher.annulus import AnnulusMesher


def get_mesher(name, **kwargs):
    return subclass_where(BaseMesher, name=name)(**kwargs)


__all__ = ["AnnulusMesher", "BaseMesher", "CubeMesher", "get_mesher"]
