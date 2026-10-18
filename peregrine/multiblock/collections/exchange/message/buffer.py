from mpi4py.MPI import Prequest, Request

from peregrine.misc import get_comm_rank_size
from peregrine.multiblock.collections.exchange.message.base import BaseMessage


class BufferMessage(BaseMessage):
    def __init__(self, backend, spool, rpool, tag):
        super().__init__(backend, spool, rpool)
        comm = get_comm_rank_size()[0]
        self.ranks = ranks = sorted(spool.slices)
        sbufs, rbufs = self.buffers()
        self.sends = [comm.Send_init(sbufs[r], r, tag) for r in ranks]
        self.recvs = [comm.Recv_init(rbufs[r], r, tag) for r in ranks]

    def buffers(self):
        raise NotImplementedError

    def expect(self):
        Prequest.Startall(self.recvs)

    def copy_out(self):
        # nothing to copy: MPI reads the buffers as they are
        pass

    def send(self):
        Prequest.Startall(self.sends)

    def recv(self):
        Request.Waitall(self.recvs)

    def sent(self):
        Request.Waitall(self.sends)
