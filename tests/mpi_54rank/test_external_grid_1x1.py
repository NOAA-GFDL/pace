import os

import numpy as np
import xarray as xr
import yaml

import pace.util
from pace.driver import Driver, DriverConfig
from pace.util.grid import MetricTerms
from pace.util.mpi import MPIComm


DIR = os.path.dirname(os.path.abspath(__file__))


def get_quantity_factory(layout, nx_tile, ny_tile, nz):
    nx = nx_tile // layout[0]
    ny = ny_tile // layout[1]
    return pace.util.QuantityFactory(
        sizer=pace.util.SubtileGridSizer(
            nx=nx, ny=ny, nz=nz, n_halo=3, extra_dim_lengths={}
        ),
        numpy=np,
    )


def get_cube_comm(layout, comm: MPIComm):
    return pace.util.CubedSphereCommunicator(
        comm=comm,
        partitioner=pace.util.CubedSpherePartitioner(
            pace.util.TilePartitioner(layout=layout)
        ),
    )


def get_tile_num(comm: MPIComm):
    return pace.util.get_tile_index(comm.rank, comm.partitioner.total_ranks)


def test_extgrid_equals_generated_1x1():

    with open(
        os.path.join(DIR, "../../driver/examples/configs/test_external_C12_2x2.yaml"),
        "r",
    ) as ext_f:
        ext_config = yaml.safe_load(ext_f)
        ext_driver_config = DriverConfig.from_dict(ext_config)

    ext_driver = Driver(ext_driver_config)

    nx_tile, ny_tile, nz = 12, 12, 5
    comm_1by1 = MPIComm()
    cube_comm = get_cube_comm(layout=(1, 1), comm=comm_1by1)

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
        global_extent=(npx, npy),
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

    metric_terms_ext = MetricTerms.from_external(
        x=lon[subtile_slice_grid],
        y=lat[subtile_slice_grid],
        dx=dx[subtile_slice_dx],
        dy=dy[subtile_slice_dy],
        area=area[subtile_slice_area],
        quantity_factory=get_quantity_factory(
            layout=(1, 1), nx_tile=nx_tile, ny_tile=ny_tile, nz=nz
        ),
        communicator=cube_comm,
        grid_type=0,
        extdgrid=True,
    )

    errors = []

    if (
        ext_driver.state.grid_data.lon.data.any()
        != metric_terms_ext.grid.data[:, :, 0].any()
    ):
        errors.append("lon data mismatch between generated and external grid data")

    if (
        ext_driver.state.grid_data.lat.data.any()
        != metric_terms_ext.grid.data[:, :, 1].any()
    ):
        errors.append("lon data mismatch between generated and external grid data")

    if (
        ext_driver.state.grid_data.dx.data.any()
        != metric_terms_ext._dx.view[:, :].any()
    ):
        errors.append("dx data mismatch between generated and external grid data")

    if (
        ext_driver.state.grid_data.dy.data.any()
        != metric_terms_ext._dy.view[:, :].any()
    ):
        errors.append("dy data mismatch between generated and external grid data")

    if (
        ext_driver.state.grid_data.area.data.any()
        != metric_terms_ext._area.view[:, :].any()
    ):
        errors.append("area data mismatch between generated and external grid data")

    assert not errors, "errors occured in 1x1:\n{}".format("\n".join(errors))