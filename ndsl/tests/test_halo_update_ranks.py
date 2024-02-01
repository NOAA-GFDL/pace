import pytest

from ndsl.comm.communicator import CubedSphereCommunicator
from ndsl.comm.partitioner import CubedSpherePartitioner, TilePartitioner
from ndsl.constants import (
    X_DIM,
    X_INTERFACE_DIM,
    Y_DIM,
    Y_INTERFACE_DIM,
    Z_DIM,
    Z_INTERFACE_DIM,
)
from ndsl.performance.timer import Timer
from ndsl.quantity import Quantity
from ndsl.testing import DummyComm


@pytest.fixture
def dtype(numpy):
    return numpy.float64


@pytest.fixture(params=[(1, 1)])
def layout(request):
    return request.param


@pytest.fixture
def ranks_per_tile(layout):
    return layout[0] * layout[1]


@pytest.fixture
def total_ranks(ranks_per_tile):
    return 6 * ranks_per_tile


@pytest.fixture
def shape(nz, ny, nx, dims, n_points):
    return_list = []
    length_dict = {
        X_DIM: 2 * n_points + nx,
        X_INTERFACE_DIM: 2 * n_points + nx + 1,
        Y_DIM: 2 * n_points + ny,
        Y_INTERFACE_DIM: 2 * n_points + ny + 1,
        Z_DIM: nz,
        Z_INTERFACE_DIM: nz + 1,
    }
    for dim in dims:
        return_list.append(length_dict[dim])
    return return_list


@pytest.fixture
def origin(n_points, dims):
    return_list = []
    origin_dict = {
        X_DIM: n_points,
        X_INTERFACE_DIM: n_points,
        Y_DIM: n_points,
        Y_INTERFACE_DIM: n_points,
        Z_DIM: 0,
        Z_INTERFACE_DIM: 0,
    }
    for dim in dims:
        return_list.append(origin_dict[dim])
    return return_list


@pytest.fixture
def extent(n_points, dims, nz, ny, nx):
    return_list = []
    extent_dict = {
        X_DIM: nx,
        X_INTERFACE_DIM: nx + 1,
        Y_DIM: ny,
        Y_INTERFACE_DIM: ny + 1,
        Z_DIM: nz,
        Z_INTERFACE_DIM: nz + 1,
    }
    for dim in dims:
        return_list.append(extent_dict[dim])
    return return_list


@pytest.fixture
def tile_partitioner(layout):
    return TilePartitioner(layout)


@pytest.fixture
def cube_partitioner(tile_partitioner):
    return CubedSpherePartitioner(tile_partitioner)


@pytest.fixture()
def communicator_list(cube_partitioner, total_ranks):
    shared_buffer = {}
    return_list = []
    for rank in range(cube_partitioner.total_ranks):
        return_list.append(
            CubedSphereCommunicator(
                comm=DummyComm(
                    rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
                ),
                partitioner=cube_partitioner,
                timer=Timer(),
            )
        )
    return return_list


@pytest.fixture
def rank_quantity_list(total_ranks, numpy, dtype):
    quantity_list = []
    for rank in range(total_ranks):
        data = numpy.empty((3, 3), dtype=dtype)
        data[:] = numpy.nan
        data[1, 1] = rank
        quantity = Quantity(
            data,
            dims=(Y_DIM, X_DIM),
            units="m",
            origin=(1, 1),
            extent=(1, 1),
        )
        quantity_list.append(quantity)
    return quantity_list


@pytest.mark.filterwarnings("ignore:invalid value encountered in remainder")
def test_correct_rank_layout(rank_quantity_list, communicator_list, subtests, numpy):
    halo_updater_list = []
    for communicator, quantity in zip(communicator_list, rank_quantity_list):
        halo_updater = communicator.start_halo_update(quantity, 1)
        halo_updater_list.append(halo_updater)
    for halo_updater in halo_updater_list:
        halo_updater.wait()
    for rank, quantity in enumerate(rank_quantity_list):
        with subtests.test(rank=rank):
            if rank % 2 == 0:
                target_data = (
                    numpy.array(
                        [
                            [numpy.nan, rank - 1, numpy.nan],
                            [rank - 2, rank, rank + 1],
                            [numpy.nan, rank + 2, numpy.nan],
                        ]
                    )
                    % 6
                )
            else:
                target_data = (
                    numpy.array(
                        [
                            [numpy.nan, rank - 2, numpy.nan],
                            [rank - 1, rank, rank + 2],
                            [numpy.nan, rank + 1, numpy.nan],
                        ]
                    )
                    % 6
                )
            numpy.testing.assert_array_equal(quantity.data, target_data)
