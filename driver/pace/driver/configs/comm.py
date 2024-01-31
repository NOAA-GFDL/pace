import abc
import dataclasses
import os
from typing import Any, ClassVar, List

import dacite

import ndsl.dsl
import ndsl.stencils
import ndsl.util
import ndsl.util.grid
from ndsl.util.comm.caching_comm import CachingCommReader, CachingCommWriter


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
class CommConfig:
    """
    Configuration for a mpi4py-style Comm object.

    Attributes:
        config: type-specific configuration
        type: type of Comm object to create, should be one of "mpi" (default),
            "write", "read", or "null_comm"
    """

    config: CreatesComm
    type: str = "mpi"
    registry: ClassVar = {}

    @classmethod
    def register(cls, type_name):
        def register_func(func):
            cls.registry[type_name] = func
            return func

        return register_func

    def get_comm(self):
        return self.config.get_comm()

    def cleanup(self, comm):
        return self.config.cleanup(comm)

    @classmethod
    def from_dict(cls, config: dict):
        config.setdefault("config", {})
        comm_type = config.get("type", "mpi")
        if comm_type not in cls.registry:
            raise ValueError(f"Unknown comm type: {comm_type}")
        else:
            creates_comm: CreatesComm = dacite.from_dict(
                data_class=cls.registry[comm_type],
                data=config["config"],
                config=dacite.Config(strict=True),
            )
            return cls(config=creates_comm, type=comm_type)


@CommConfig.register("mpi")
@dataclasses.dataclass
class MPICommConfig(CreatesComm):
    """
    Configuration for a true mpi4py Comm object.
    """

    def get_comm(self):
        return ndsl.util.MPIComm()

    def cleanup(self, comm):
        pass


@CommConfig.register("null_comm")
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
    fill_value: float

    def get_comm(self):
        return ndsl.util.NullComm(
            rank=self.rank, total_ranks=self.total_ranks, fill_value=self.fill_value
        )

    def cleanup(self, comm):
        pass


@CommConfig.register("write")
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

    ranks: List[int]
    path: str = "."

    def get_comm(self) -> CachingCommWriter:
        underlying = MPICommConfig().get_comm()
        if underlying.Get_rank() in self.ranks:
            return ndsl.util.CachingCommWriter(underlying)
        else:
            return underlying

    def cleanup(self, comm: CachingCommWriter):
        os.makedirs(self.path, exist_ok=True)
        if comm.Get_rank() in self.ranks:
            with open(
                os.path.join(self.path, f"comm_{comm.Get_rank()}.pkl"), "wb"
            ) as f:
                comm.dump(f)


@CommConfig.register("read")
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
            return ndsl.util.CachingCommReader.load(f)

    def cleanup(self, comm: CachingCommWriter):
        pass
