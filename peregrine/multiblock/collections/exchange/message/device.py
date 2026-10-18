from mpi4py import MPI
from mpi4py.util.dtlib import from_numpy_dtype

from peregrine.multiblock.collections.exchange.message.buffer import BufferMessage


# MPI reads and writes the pools' device memory as it is, so on a device backend the MPI must be GPU-aware
class DeviceMessage(BufferMessage):
    name = "device"

    def __init__(self, backend, spool, rpool, tag):
        super().__init__(backend, spool, rpool, tag)

    def buffers(self):
        sbufs = {r: self._message(part) for r, part in self.spool.parts.items()}
        rbufs = {r: self._message(part) for r, part in self.rpool.parts.items()}
        return sbufs, rbufs

    @staticmethod
    def _message(part):
        mem = MPI.memory.fromaddress(part.ptr, part.nbytes)
        return [mem, from_numpy_dtype(part.dtype)]

    def copy_out(self):
        if self.ranks:
            self.backend.copies_await_kernels()

    def send(self):
        if self.ranks:
            self.backend.await_copies()
        super().send()
