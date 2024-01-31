import ndsl.util
from ndsl.util.comm.null_comm import NullComm


def test_can_create_cube_communicator():
    rank = 2
    total_ranks = 24
    mpi_comm = NullComm(rank, total_ranks)
    layout = (2, 2)
    partitioner = ndsl.util.CubedSpherePartitioner(ndsl.util.TilePartitioner(layout))
    communicator = ndsl.util.CubedSphereCommunicator(mpi_comm, partitioner)
    communicator.tile.partitioner
