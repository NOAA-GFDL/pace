try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from typing import List, Optional, TypeVar, cast

from ndsl.comm.comm_abc import Comm, Request
from ndsl.logging import ndsl_log


T = TypeVar("T")


class MPIComm(Comm):
    def __init__(self):
        if MPI is None:
            raise RuntimeError("MPI not available")
        self._comm: Comm = cast(Comm, MPI.COMM_WORLD)

    def Get_rank(self) -> int:
        return self._comm.Get_rank()

    def Get_size(self) -> int:
        return self._comm.Get_size()

    def bcast(self, value: Optional[T], root=0) -> T:
        ndsl_log.debug("bcast from root %s on rank %s", root, self._comm.Get_rank())
        return self._comm.bcast(value, root=root)

    def barrier(self):
        ndsl_log.debug("barrier on rank %s", self._comm.Get_rank())
        self._comm.barrier()

    def Barrier(self):
        pass

    def Scatter(self, sendbuf, recvbuf, root=0, **kwargs):
        ndsl_log.debug("Scatter on rank %s with root %s", self._comm.Get_rank(), root)
        self._comm.Scatter(sendbuf, recvbuf, root=root, **kwargs)

    def Gather(self, sendbuf, recvbuf, root=0, **kwargs):
        ndsl_log.debug("Gather on rank %s with root %s", self._comm.Get_rank(), root)
        self._comm.Gather(sendbuf, recvbuf, root=root, **kwargs)

    def allgather(self, sendobj: T) -> List[T]:
        ndsl_log.debug("allgather on rank %s", self._comm.Get_rank())
        return self._comm.allgather(sendobj)

    def Send(self, sendbuf, dest, tag: int = 0, **kwargs):
        ndsl_log.debug("Send on rank %s with dest %s", self._comm.Get_rank(), dest)
        self._comm.Send(sendbuf, dest, tag=tag, **kwargs)

    def sendrecv(self, sendbuf, dest, **kwargs):
        ndsl_log.debug("sendrecv on rank %s with dest %s", self._comm.Get_rank(), dest)
        return self._comm.sendrecv(sendbuf, dest, **kwargs)

    def Isend(self, sendbuf, dest, tag: int = 0, **kwargs) -> Request:
        ndsl_log.debug("Isend on rank %s with dest %s", self._comm.Get_rank(), dest)
        return self._comm.Isend(sendbuf, dest, tag=tag, **kwargs)

    def Recv(self, recvbuf, source, tag: int = 0, **kwargs):
        ndsl_log.debug("Recv on rank %s with source %s", self._comm.Get_rank(), source)
        self._comm.Recv(recvbuf, source, tag=tag, **kwargs)

    def Irecv(self, recvbuf, source, tag: int = 0, **kwargs) -> Request:
        ndsl_log.debug("Irecv on rank %s with source %s", self._comm.Get_rank(), source)
        return self._comm.Irecv(recvbuf, source, tag=tag, **kwargs)

    def Split(self, color, key) -> "Comm":
        ndsl_log.debug(
            "Split on rank %s with color %s, key %s", self._comm.Get_rank(), color, key
        )
        return self._comm.Split(color, key)

    def allreduce(self, sendobj: T, op=None) -> T:
        ndsl_log.debug(
            "allreduce on rank %s with operator %s", self._comm.Get_rank(), op
        )
        return self._comm.allreduce(sendobj, op)
