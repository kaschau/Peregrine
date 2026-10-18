from mpi4py import MPI


def get_comm_rank_size():
    comm = MPI.COMM_WORLD
    return comm, comm.rank, comm.size


def get_local_rank():
    node = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    rank = node.rank
    node.Free()
    return rank


def get_ranks_per_node():
    comm, rank, size = get_comm_rank_size()
    node = comm.Split_type(MPI.COMM_TYPE_SHARED)
    # the max so every rank agrees when the last node is only partly filled
    ranks_per_node = comm.allreduce(node.size, op=MPI.MAX)
    node.Free()
    return ranks_per_node
