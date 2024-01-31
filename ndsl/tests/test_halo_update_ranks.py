import pytest

import ndsl.util


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
        ndsl.util.X_DIM: 2 * n_points + nx,
        ndsl.util.X_INTERFACE_DIM: 2 * n_points + nx + 1,
        ndsl.util.Y_DIM: 2 * n_points + ny,
        ndsl.util.Y_INTERFACE_DIM: 2 * n_points + ny + 1,
        ndsl.util.Z_DIM: nz,
        ndsl.util.Z_INTERFACE_DIM: nz + 1,
    }
    for dim in dims:
        return_list.append(length_dict[dim])
    return return_list


@pytest.fixture
def origin(n_points, dims):
    return_list = []
    origin_dict = {
        ndsl.util.X_DIM: n_points,
        ndsl.util.X_INTERFACE_DIM: n_points,
        ndsl.util.Y_DIM: n_points,
        ndsl.util.Y_INTERFACE_DIM: n_points,
        ndsl.util.Z_DIM: 0,
        ndsl.util.Z_INTERFACE_DIM: 0,
    }
    for dim in dims:
        return_list.append(origin_dict[dim])
    return return_list


@pytest.fixture
def extent(n_points, dims, nz, ny, nx):
    return_list = []
    extent_dict = {
        ndsl.util.X_DIM: nx,
        ndsl.util.X_INTERFACE_DIM: nx + 1,
        ndsl.util.Y_DIM: ny,
        ndsl.util.Y_INTERFACE_DIM: ny + 1,
        ndsl.util.Z_DIM: nz,
        ndsl.util.Z_INTERFACE_DIM: nz + 1,
    }
    for dim in dims:
        return_list.append(extent_dict[dim])
    return return_list


@pytest.fixture
def tile_partitioner(layout):
    return ndsl.util.TilePartitioner(layout)


@pytest.fixture
def cube_partitioner(tile_partitioner):
    return ndsl.util.CubedSpherePartitioner(tile_partitioner)


@pytest.fixture()
def communicator_list(cube_partitioner, total_ranks):
    shared_buffer = {}
    return_list = []
    for rank in range(cube_partitioner.total_ranks):
        return_list.append(
            ndsl.util.CubedSphereCommunicator(
                comm=ndsl.util.testing.DummyComm(
                    rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
                ),
                partitioner=cube_partitioner,
                timer=ndsl.util.Timer(),
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
        quantity = ndsl.util.Quantity(
            data,
            dims=(ndsl.util.Y_DIM, ndsl.util.X_DIM),
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
