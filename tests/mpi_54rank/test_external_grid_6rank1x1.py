import numpy as np
import xarray as xr

import pace.driver
import pace.util
from pace.util.grid import MetricTerms
from pace.util.mpi import MPIComm


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


def test_extgrid_equals_generated_1x1_6():
    nx_tile, ny_tile, nz = 6, 6, 5
    comm_1by1 = MPIComm()
    cube_comm = get_cube_comm(layout=(1, 1), comm=comm_1by1)
    metric_terms_gen = MetricTerms(
        quantity_factory=get_quantity_factory(
            layout=(1, 1), nx_tile=nx_tile, ny_tile=ny_tile, nz=nz
        ),
        communicator=cube_comm,
    )

    metric_terms_gen._dx, metric_terms_gen._dy = metric_terms_gen._compute_dxdy()

    tile_num = get_tile_num(cube_comm) + 1
    tile_file = "../../test_input/ext_test_6.tile" + str(tile_num) + ".nc"
    ds = xr.open_dataset(tile_file)
    lon = ds.x.values
    lat = ds.y.values
    dx = ds.dx.values
    dy = ds.dy.values
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
        global_extent=(npx, npy),
        overlap=True,
    )

    subtile_slice_dy = cube_comm.partitioner.tile.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.Y_DIM, pace.util.X_INTERFACE_DIM],
        global_extent=(npx, npy),
        overlap=True,
    )

    metric_terms_ext = MetricTerms.from_external(
        x=lon[subtile_slice_grid],
        y=lat[subtile_slice_grid],
        dx=dx[subtile_slice_dx],
        dy=dy[subtile_slice_dy],
        quantity_factory=get_quantity_factory(
            layout=(1, 1), nx_tile=nx_tile, ny_tile=ny_tile, nz=nz
        ),
        communicator=cube_comm,
        grid_type=0,
        extdgrid=True,
    )

    print(metric_terms_ext._dx)

    errors = []

    if (
        metric_terms_gen.grid.data[:, :, 0].any()
        != metric_terms_ext.grid.data[:, :, 0].any()
    ):
        errors.append("lon data mismatch between generated and external grid data")

    if (
        metric_terms_gen.grid.data[:, :, 1].any()
        != metric_terms_ext.grid.data[:, :, 1].any()
    ):
        errors.append("lat data mismatch between generated and external grid data")

    if metric_terms_gen._dx.view[:, :].any() != metric_terms_ext._dx.view[:, :].any():
        errors.append("dx data mismatch between generated and external grid data")

    if metric_terms_gen._dy.view[:, :].any() != metric_terms_ext._dy.view[:, :].any():
        errors.append("dy data mismatch between generated and external grid data")

    # if metric_terms_gen._area.data[:,:].any() != metric_terms_ext._area.any():
    #    errors.append("area data mismatch between generated and external grid data")

    assert not errors, "errors occured in 1x1_6:\n{}".format("\n".join(errors))
