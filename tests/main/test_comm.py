from ndsl import CubedSphereCommunicator, CubedSpherePartitioner, TilePartitioner
from pace import NullComm


def test_can_create_cube_communicator():
    null_comm = NullComm(rank=2, total_ranks=24)
    partitioner = CubedSpherePartitioner(TilePartitioner(layout=(2, 2)))
    communicator = CubedSphereCommunicator(null_comm, partitioner)

    assert communicator.tile.partitioner
