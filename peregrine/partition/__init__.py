from peregrine.misc import subclass_where
from peregrine.partition.cut import CutMixin
from peregrine.partition.merge import MergeMixin
from peregrine.partition.orient import OrientMixin
from peregrine.partition.conditioner import Conditioner
from peregrine.partition.base import BasePartitioner
from peregrine.partition.greedy import GreedyPartitioner
from peregrine.partition.metis import MetisPartitioner
from peregrine.partition.auto import AutoPartitioner


def get_partitioner(name="auto", **kwargs):
    return subclass_where(BasePartitioner, name=name)(**kwargs)


__all__ = [
    "AutoPartitioner",
    "BasePartitioner",
    "CutMixin",
    "Conditioner",
    "GreedyPartitioner",
    "MergeMixin",
    "MetisPartitioner",
    "OrientMixin",
    "get_partitioner",
]
