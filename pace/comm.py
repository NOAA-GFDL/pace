import abc
import copy
import dataclasses
import os
from typing import Any, ClassVar, Mapping, TypeVar, cast

from ndsl import MPIComm
from ndsl.comm import (
    CachingCommReader,
    CachingCommWriter,
    Comm,
    ReductionOperator,
    Request,
)
from pace.registry import Registry


T = TypeVar("T")


class NullAsyncResult(Request):
    def __init__(self, recvbuf: Any = None) -> None:
        self._recvbuf = recvbuf

    def wait(self) -> None:
        if self._recvbuf is not None:
            self._recvbuf[:] = 0.0


class NullComm(Comm[T]):
    """
    A class with a subset of the mpi4py Comm API, but which
    'receives' a fill value (default zero) instead of using MPI.
    """

    default_fill_value: T = cast(T, 0)

    def __init__(self, rank: int, total_ranks: int, fill_value: T = default_fill_value):
        """
        Args:
            rank: rank to mock
            total_ranks: number of total MPI ranks to mock
            fill_value: fill halos with this value when performing
                halo updates.
        """
        self.rank = rank
        self.total_ranks = total_ranks
        self._fill_value = fill_value
        self._split_comms: Mapping[Any, list[NullComm]] = {}

    def __repr__(self) -> str:
        return f"NullComm(rank={self.rank}, total_ranks={self.total_ranks})"

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return self.total_ranks

    def bcast(self, value: T | None, root: int = 0) -> T | None:
        return value

    def barrier(self) -> None:
        return

    def Barrier(self) -> None:
        return

    def Scatter(self, sendbuf, recvbuf, root: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        if recvbuf is not None:
            recvbuf[:] = self._fill_value

    def Gather(self, sendbuf, recvbuf, root: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        if recvbuf is not None:
            recvbuf[:] = self._fill_value

    def allgather(self, sendobj: T) -> list[T]:
        return [copy.deepcopy(sendobj) for _ in range(self.total_ranks)]

    def Send(self, sendbuf, dest, tag: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        pass

    def Isend(self, sendbuf, dest, tag: int = 0, **kwargs: dict) -> Request:  # type: ignore[no-untyped-def]
        return NullAsyncResult()

    def Recv(self, recvbuf, source, tag: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        recvbuf[:] = self._fill_value

    def Irecv(self, recvbuf, source, tag: int = 0, **kwargs: dict) -> Request:  # type: ignore[no-untyped-def]
        return NullAsyncResult(recvbuf)

    def sendrecv(self, sendbuf, dest, **kwargs: dict):  # type: ignore[no-untyped-def]
        return sendbuf

    def Split(self, color, key) -> Comm:  # type: ignore[no-untyped-def]
        # key argument is ignored, assumes we're calling the ranks from least to
        # greatest when mocking Split
        self._split_comms[color] = self._split_comms.get(color, [])  # type: ignore[index]
        rank = len(self._split_comms[color])
        total_ranks = rank + 1
        new_comm = NullComm(
            rank=rank, total_ranks=total_ranks, fill_value=self._fill_value
        )
        for comm in self._split_comms[color]:
            # won't know how many ranks there are until everything is split
            comm.total_ranks = total_ranks
        self._split_comms[color].append(new_comm)
        return new_comm

    def allreduce(
        self, sendobj: T, op: ReductionOperator = ReductionOperator.NO_OP
    ) -> T:
        return self._fill_value

    def Allreduce(self, sendobj: T, recvobj: T, op: ReductionOperator) -> T:
        # TODO: what about reduction operator `op`?
        recvobj = sendobj
        return recvobj

    def Allreduce_inplace(self, obj: T, op: ReductionOperator) -> T:
        raise NotImplementedError("NullComm.Allreduce_inplace")
    
    def Scatterv(self, sendbuf, recvbuf, root: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        pass


class CreatesComm(abc.ABC):
    """
    Retrieves and does cleanup for a mpi4py-style Comm object.
    """

    @abc.abstractmethod
    def get_comm(self) -> Any:
        """
        Get an mpi4py-style Comm object.
        """
        ...

    @abc.abstractmethod
    def cleanup(self, comm):
        """
        Perform any operations that must occur before exiting.
        """
        ...


@dataclasses.dataclass(frozen=True)
class CreatesCommSelector(CreatesComm):
    """
    Dataclass for selecting the CreatesComm implementation to use.

    Used to circumvent the issue that dacite expects static class definitions,
    but we would like to dynamically define which CreatesComm to use. Does this
    by representing the part of the yaml specification that asks which comm creator
    to use, but deferring to the implementation in that selected type when called.

    Attributes:
        config: type-specific configuration
        type: type of Comm object to create, should be one of "mpi" (default),
            "write", "read", or "null_comm"
    """

    config: CreatesComm = dataclasses.field(default_factory=lambda: MPICommConfig())
    type: str = "mpi"
    registry: ClassVar[Registry] = Registry(default_type="mpi")

    @classmethod
    def register(cls, type_name):
        return cls.registry.register(type_name)

    def get_comm(self) -> Comm:
        """
        Get an mpi4py-style Comm object.

        Returns:
            comm: an mpi4py-style Comm object
        """
        return self.config.get_comm()

    def cleanup(self, comm):
        return self.config.cleanup(comm)

    @classmethod
    def from_dict(cls, config: dict):
        creates_comm = cls.registry.from_dict(config)
        return cls(
            config=creates_comm, type=config.get("type", cls.registry.default_type)
        )


@CreatesCommSelector.register("mpi")
@dataclasses.dataclass
class MPICommConfig(CreatesComm):
    """
    Configuration for a true mpi4py Comm object.
    """

    def get_comm(self):
        return MPIComm()

    def cleanup(self, comm):
        pass


@CreatesCommSelector.register("null_comm")
@dataclasses.dataclass
class NullCommConfig(CreatesComm):
    """
    Configuration for a NullComm object which does not perform halo updates,
    instead filling the halos with a constant value.

    Generally used to test whether the code crashes while running in serial when
    correctness of the answer is not important.

    Attributes:
        rank: rank of the comm
        total_ranks: the total number of ranks for the comm to pretend to have
        fill_value: the value to fill the halos with
    """

    rank: int
    total_ranks: int
    fill_value: float = 0.0

    def get_comm(self):
        return NullComm(
            rank=self.rank, total_ranks=self.total_ranks, fill_value=self.fill_value
        )

    def cleanup(self, comm):
        pass


@CreatesCommSelector.register("write")
@dataclasses.dataclass
class WriterCommConfig(CreatesComm):
    """
    Configuration for a CachingCommWriter object.

    This object will wrap a real mpi4py comm object, but will cache the
    communication between the ranks in the comm and write the result to disk
    at cleanup.

    This data can later be read in a run using a ReaderCommConfig.

    Attributes:
        ranks: which ranks to write data for
        path: directory to write data to
    """

    ranks: list[int]
    path: str = "."

    def get_comm(self) -> CachingCommWriter:
        underlying = MPICommConfig().get_comm()
        if underlying.Get_rank() in self.ranks:
            return CachingCommWriter(underlying)
        else:
            return underlying

    def cleanup(self, comm: CachingCommWriter):
        os.makedirs(self.path, exist_ok=True)
        if comm.Get_rank() in self.ranks:
            with open(
                os.path.join(self.path, f"comm_{comm.Get_rank()}.pkl"), "wb"
            ) as f:
                comm.dump(f)


@CreatesCommSelector.register("read")
@dataclasses.dataclass
class ReaderCommConfig(CreatesComm):
    """
    Configuration for a CachingCommReader object.

    This object reads data cached by a WriterCommConfig, and will perform
    identical communication as was written by that writer, played back
    in the same order.

    This should generally be used within an identical configuration as was used by
    the WriterCommConfig, and must be used with a configuration that will result
    in an identical communication pattern.

    Attributes:
        rank: rank to read data for
        path: directory to read data from
    """

    rank: int
    path: str = "."

    def get_comm(self) -> CachingCommReader:
        with open(os.path.join(self.path, f"comm_{self.rank}.pkl"), "rb") as f:
            return CachingCommReader.load(f)

    def cleanup(self, comm: CachingCommWriter):
        pass
