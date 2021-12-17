import pytest

import pace.util
from fv3core.grid import MetricTerms
from fv3core.utils.null_comm import NullComm


@pytest.mark.parametrize(
    "rank, edge_interior_ratio",
    [
        pytest.param(
            0,
            1.0,
            id="rank_0_ratio_1.0",
        ),
        pytest.param(
            0,
            0.9,
            id="rank_0_ratio_0.9",
        ),
        pytest.param(
            5,
            0.9,
            id="rank_5_ratio_0.9",
        ),
        pytest.param(
            0,
            0.3,
            id="rank_0_ratio_0.3",
        ),
    ],
)
def test_metric_terms(rank: int, edge_interior_ratio: float):
    layout = [3, 3]
    npx, npy, npz = 129, 129, 79
    total_ranks = layout[0] * layout[1]
    backend = "numpy"

    mpi_comm = NullComm(rank, total_ranks)
    partitioner = pace.util.CubedSpherePartitioner(
        pace.util.TilePartitioner(layout, edge_interior_ratio)
    )
    communicator = pace.util.CubedSphereCommunicator(mpi_comm, partitioner)

    terms = MetricTerms.from_tile_sizing(
        npx=npx, npy=npy, npz=npz, communicator=communicator, backend=backend
    )
