from mpi4py.MPI import Request

from peregrine.multiblock.collections.exchange.message.buffer import BufferMessage


class HostStagedMessage(BufferMessage):
    name = "hoststaged"

    def __init__(self, backend, spool, rpool, tag):
        super().__init__(backend, spool, rpool, tag)

    def buffers(self):
        backend, spool, rpool = self.backend, self.spool, self.rpool
        self.smirror = backend.pinned(spool.array.shape, spool.array.dtype)
        self.rmirror = backend.pinned(rpool.array.shape, rpool.array.dtype)
        sbufs = {r: self.smirror[sl] for r, sl in spool.slices.items()}
        rbufs = {r: self.rmirror[sl] for r, sl in rpool.slices.items()}
        return sbufs, rbufs

    # the receives land in rmirror, which the last recv's copies out of it may still be reading
    def expect(self):
        if self.ranks:
            self.backend.await_copies()
        super().expect()

    def copy_out(self):
        if self.ranks:
            self.spool.array.async_pull(self.smirror)

    def send(self):
        if self.ranks:
            self.backend.await_copies()
        super().send()

    def recv(self):
        if not self.ranks:
            return
        # a landed persistent request goes inactive and Waitany passes it over, so each pass picks a message not yet copied in
        for _ in self.ranks:
            r = self.ranks[Request.Waitany(self.recvs)]
            self.rpool.parts[r].async_push(self.rmirror[self.rpool.slices[r]])
        self.backend.kernels_await_copies()
