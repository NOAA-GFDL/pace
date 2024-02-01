from ndsl.comm.communicator import CubedSphereCommunicator
from ndsl.comm.null_comm import NullComm
from ndsl.comm.partitioner import CubedSpherePartitioner, TilePartitioner


def test_can_create_cube_communicator():
    rank = 2
    total_ranks = 24
    mpi_comm = NullComm(rank, total_ranks)
    layout = (2, 2)
    partitioner = CubedSpherePartitioner(TilePartitioner(layout))
    communicator = CubedSphereCommunicator(mpi_comm, partitioner)
    communicator.tile.partitioner
