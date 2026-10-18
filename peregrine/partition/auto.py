from peregrine.partition.base import BasePartitioner
from peregrine.partition.greedy import GreedyPartitioner
from peregrine.partition.metis import MetisPartitioner


class AutoPartitioner(BasePartitioner):
    name = "auto"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metis = MetisPartitioner(**kwargs)
        self.greedy = GreedyPartitioner(**kwargs)

    def _assign(self, weights, edges, k):
        # metis balances to a target, not a cap, so it only pays once every rank holds many blocks
        manyblks = weights.shape[0] >= 16 * k
        kind = self.metis if manyblks and edges else self.greedy
        return kind._assign(weights, edges, k)
