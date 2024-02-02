import numpy as np
import pytest
from ndsl.comm.communicator import CubedSphereCommunicator
from ndsl.comm.null_comm import NullComm
from ndsl.comm.partitioner import CubedSpherePartitioner, TilePartitioner
from ndsl.grid import MetricTerms
from ndsl.initialization.allocator import QuantityFactory
from ndsl.initialization.sizer import SubtileGridSizer
from ndsl.quantity import Quantity


def get_cube_comm(layout, rank: int):
    return CubedSphereCommunicator(
        comm=NullComm(rank=rank, total_ranks=6 * layout[0] * layout[1]),
        partitioner=CubedSpherePartitioner(TilePartitioner(layout=layout)),
    )


def get_quantity_factory(layout, nx_tile, ny_tile, nz):
    nx = nx_tile // layout[0]
    ny = ny_tile // layout[1]
    return QuantityFactory(
        sizer=SubtileGridSizer(nx=nx, ny=ny, nz=nz, n_halo=3, extra_dim_lengths={}),
        numpy=np,
    )


@pytest.mark.parametrize("rank", [0, 1, 4])
def test_grid_init_not_decomposition_dependent(rank: int):
    """
    This is a limited version of the full grid and state init tests in the
    mpi_54rank test suite. It tests only variables that are not dependent on
    halo updates for their values in the compute domain.
    """
    nx_tile, ny_tile, nz = 48, 48, 79
    eta_file = "tests/main/input/eta79.nc"
    metric_terms_1by1 = MetricTerms(
        quantity_factory=get_quantity_factory(
            layout=(1, 1), nx_tile=nx_tile, ny_tile=ny_tile, nz=nz
        ),
        communicator=get_cube_comm(rank=0, layout=(1, 1)),
        eta_file=eta_file,
    )
    metric_terms_3by3 = MetricTerms(
        quantity_factory=get_quantity_factory(
            layout=(3, 3), nx_tile=nx_tile, ny_tile=ny_tile, nz=nz
        ),
        communicator=get_cube_comm(rank=rank, layout=(3, 3)),
        eta_file=eta_file,
    )
    partitioner = TilePartitioner(layout=(3, 3))
    assert allclose(metric_terms_1by1.grid, metric_terms_3by3.grid, partitioner, rank)
    assert allclose(metric_terms_1by1.agrid, metric_terms_3by3.agrid, partitioner, rank)
    assert allclose(metric_terms_1by1.area, metric_terms_3by3.area, partitioner, rank)
    assert allclose(metric_terms_1by1.dx, metric_terms_3by3.dx, partitioner, rank)
    assert allclose(metric_terms_1by1.dy, metric_terms_3by3.dy, partitioner, rank)
    assert allclose(
        metric_terms_1by1.cos_sg1, metric_terms_3by3.cos_sg1, partitioner, rank
    )
    assert allclose(
        metric_terms_1by1.cos_sg2, metric_terms_3by3.cos_sg2, partitioner, rank
    )
    assert allclose(
        metric_terms_1by1.cos_sg3, metric_terms_3by3.cos_sg3, partitioner, rank
    )
    assert allclose(
        metric_terms_1by1.cos_sg4, metric_terms_3by3.cos_sg4, partitioner, rank
    )
    assert allclose(
        metric_terms_1by1.sin_sg1, metric_terms_3by3.sin_sg1, partitioner, rank
    )
    assert allclose(
        metric_terms_1by1.sin_sg2, metric_terms_3by3.sin_sg2, partitioner, rank
    )
    assert allclose(
        metric_terms_1by1.sin_sg3, metric_terms_3by3.sin_sg3, partitioner, rank
    )
    assert allclose(
        metric_terms_1by1.sin_sg4, metric_terms_3by3.sin_sg4, partitioner, rank
    )
    assert allclose(metric_terms_1by1.rarea, metric_terms_3by3.rarea, partitioner, rank)
    assert allclose(metric_terms_1by1.rdx, metric_terms_3by3.rdx, partitioner, rank)
    assert allclose(metric_terms_1by1.rdy, metric_terms_3by3.rdy, partitioner, rank)


def allclose(
    q_1by1: Quantity,
    q_3by3: Quantity,
    partitioner: TilePartitioner,
    rank: int,
):
    subtile_slice = partitioner.subtile_slice(
        rank=rank, global_dims=q_1by1.dims, global_extent=q_1by1.extent, overlap=True
    )
    v1 = q_1by1.view[subtile_slice]
    v2 = q_3by3.view[:]
    assert v1.shape == v2.shape
    same = (v1 == v2) | np.isnan(v1)
    all_same = np.all(same)
    if not all_same:
        print(np.sum(~same), np.where(~same))
    return all_same
