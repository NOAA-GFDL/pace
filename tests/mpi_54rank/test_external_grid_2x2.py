import math
import os

import numpy as np
import xarray as xr
import yaml

import pace.util
from pace.driver import Driver, DriverConfig
from pace.util.constants import PI, RADIUS
from pace.util.mpi import MPIComm


DIR = os.path.dirname(os.path.abspath(__file__))


def get_cube_comm(layout, comm: MPIComm):
    return pace.util.CubedSphereCommunicator(
        comm=comm,
        partitioner=pace.util.CubedSpherePartitioner(
            pace.util.TilePartitioner(layout=layout)
        ),
    )


def get_tile_num(comm: MPIComm):
    return pace.util.get_tile_index(comm.rank, comm.partitioner.total_ranks)


def test_extgrid_equals_generated_2x2():

    with open(
        os.path.join(DIR, "../../driver/examples/configs/test_external_C12_2x2.yaml"),
        "r",
    ) as ext_f:
        ext_config = yaml.safe_load(ext_f)
        ext_driver_config = DriverConfig.from_dict(ext_config)

    ext_driver = Driver(ext_driver_config)

    comm_2by2 = MPIComm()
    cube_comm = get_cube_comm(layout=(2, 2), comm=comm_2by2)

    tile_num = get_tile_num(cube_comm) + 1
    tile_file = "../../test_input/C12.tile" + str(tile_num) + ".nc"
    ds = xr.open_dataset(os.path.join(DIR, tile_file))
    lon = ds.x.values
    lat = ds.y.values
    dx = ds.dx.values
    dy = ds.dy.values
    area = ds.area.values
    nx = ds.nx.values.size
    ny = ds.ny.values.size
    npx = ds.nxp.values.size
    npy = ds.nyp.values.size

    subtile_slice_grid = cube_comm.partitioner.tile.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.Y_INTERFACE_DIM, pace.util.X_INTERFACE_DIM],
        global_extent=(npy, npx),
        overlap=True,
    )

    subtile_slice_dx = cube_comm.partitioner.tile.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.Y_INTERFACE_DIM, pace.util.X_DIM],
        global_extent=(npy, nx),
        overlap=True,
    )

    subtile_slice_dy = cube_comm.partitioner.tile.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.Y_DIM, pace.util.X_INTERFACE_DIM],
        global_extent=(ny, npx),
        overlap=True,
    )

    subtile_slice_area = cube_comm.partitioner.tile.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.Y_DIM, pace.util.X_DIM],
        global_extent=(ny, nx),
        overlap=True,
    )

    lon_rad = lon * (PI / 180)
    lat_rad = lat * (PI / 180)

    errors = []

    if not np.isclose(
        ext_driver.state.grid_data.lon.view[:, :], lon_rad[subtile_slice_grid]
    ).all():
        errors.append("Lon data mismatch")

    if not np.isclose(
        ext_driver.state.grid_data.lat.view[:, :], lat_rad[subtile_slice_grid]
    ).all():
        errors.append("Lat data mismatch")

    if not np.isclose(
        ext_driver.state.grid_data.dy.view[:, :], dx[subtile_slice_dx]
    ).all():
        errors.append("dx data mismatch")

    if not np.isclose(
        ext_driver.state.grid_data.dx.view[:, :], dy[subtile_slice_dy]
    ).all():
        errors.append("dy data mismatch")

    if not np.isclose(
        ext_driver.state.grid_data.area.view[:, :], area[subtile_slice_area]
    ).all():
        errors.append("area data mismatch")

    assert not errors, "errors occured in 2x2:\n{}".format("\n".join(errors))

    surface_area_true = 4 * PI * (RADIUS ** 2)

    mpicomm = MPIComm()

    tmp = [math.fsum(row) for row in ext_driver.state.grid_data.area.view[:, :]]
    rank_area_sum = math.fsum(tmp)
    tile_area = mpicomm._comm.gather(rank_area_sum, root=0)

    if mpicomm.Get_rank() == 0:
        # print(ext_driver.state.grid_data.dy.view[:, :])
        # print("")
        # print(dx[subtile_slice_dx])
        total_area = math.fsum(tile_area)
        assert np.isclose(total_area, surface_area_true)
