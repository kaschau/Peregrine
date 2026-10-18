import numpy as np

from peregrine.partition.base import BasePartitioner


class MetisPartitioner(BasePartitioner):
    name = "metis"

    def _assign(self, weights, edges, k):
        # pymetis only for this partitioner
        import pymetis

        adj = [[] for _ in range(weights.shape[0])]
        for (a, b), w in edges.items():
            adj[a].append((b, w))
            adj[b].append((a, w))
        xadj, adjncy, eweights = [0], [], []
        for a in range(weights.shape[0]):
            for b, w in adj[a]:
                adjncy.append(b)
                eweights.append(int(w))
            xadj.append(len(adjncy))

        opts = pymetis.Options()
        # metis takes imbalance in thousandths, as a target rather than a cap
        opts.ufactor = max(1, int(self.tol * 1000))
        opts.seed = 0
        result = pymetis.part_graph(
            k,
            adjacency=pymetis.CSRAdjacency(xadj, adjncy),
            vweights=[int(w) for w in weights],
            eweights=eweights,
            options=opts,
        )
        return np.array(result.vertex_part, dtype=np.int64)
