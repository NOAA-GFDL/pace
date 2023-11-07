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

    tile_file = "/input/tilefile"
    ds = xr.open_dataset(tile_file)
    lon = ds.x.values
    lat = ds.y.values
    dx = ds.dx.values
    dy = ds.dy.values
    npx = ds.nxp.values.size
    npy = ds.nyp.values.size

    subtile_slice_grid = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_INTERFACE_DIM, pace.util.Y_INTERFACE_DIM],
        global_extent=(npx, npy),
        overlap=True,
    )

    subtile_slice_dx = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_INTERFACE_DIM, pace.util.Y_DIM],
        global_extent=(npx - 1, npy),
        overlap=True,
    )

    subtile_slice_dy = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_DIM, pace.util.Y_INTERFACE_DIM],
        global_extent=(npx, npy - 1),
        overlap=True,
    )

    metric_terms_ext = MetricTerms.from_generated(
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

    errors = []

    if metric_terms_gen.grid.data[:, :, 0] != metric_terms_ext.grid.data[:, :, 0]:
        errors.append("lon data mismatch between generated and external grid data")

    if metric_terms_gen.grid.data[:, :, 1] != metric_terms_ext.grid.data[:, :, 1]:
        errors.append("lat data mismatch between generated and external grid data")

    if metric_terms_gen._dx != metric_terms_ext._dx:
        errors.append("dx data mismatch between generated and external grid data")

    if metric_terms_gen._dy != metric_terms_ext._dy:
        errors.append("dy data mismatch between generated and external grid data")

    if metric_terms_gen._area != metric_terms_ext._area:
        errors.append("area data mismatch between generated and external grid data")

    assert not errors, "errors occured in 1x1_6:\n{}".format("\n".join(errors))


def test_extgrid_equals_generated_1x1_24():
    nx_tile, ny_tile, nz = 24, 24, 5
    comm_1by1 = MPIComm()
    cube_comm = get_cube_comm(layout=(1, 1), comm=comm_1by1)
    metric_terms_gen = MetricTerms(
        quantity_factory=get_quantity_factory(
            layout=(1, 1), nx_tile=nx_tile, ny_tile=ny_tile, nz=nz
        ),
        communicator=cube_comm,
    )

    tile_file = "/input/tilefile"
    ds = xr.open_dataset(tile_file)
    lon = ds.x.values
    lat = ds.y.values
    dx = ds.dx.values
    dy = ds.dy.values
    npx = ds.nxp.values.size
    npy = ds.nyp.values.size

    subtile_slice_grid = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_INTERFACE_DIM, pace.util.Y_INTERFACE_DIM],
        global_extent=(npx, npy),
        overlap=True,
    )

    subtile_slice_dx = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_INTERFACE_DIM, pace.util.Y_DIM],
        global_extent=(npx - 1, npy),
        overlap=True,
    )

    subtile_slice_dy = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_DIM, pace.util.Y_INTERFACE_DIM],
        global_extent=(npx, npy - 1),
        overlap=True,
    )

    metric_terms_ext = MetricTerms.from_generated(
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

    errors = []

    if metric_terms_gen.grid.data[:, :, 0] != metric_terms_ext.grid.data[:, :, 0]:
        errors.append("lon data mismatch between generated and external grid data")

    if metric_terms_gen.grid.data[:, :, 1] != metric_terms_ext.grid.data[:, :, 1]:
        errors.append("lat data mismatch between generated and external grid data")

    if metric_terms_gen._dx != metric_terms_ext._dx:
        errors.append("dx data mismatch between generated and external grid data")

    if metric_terms_gen._dy != metric_terms_ext._dy:
        errors.append("dy data mismatch between generated and external grid data")

    if metric_terms_gen._area != metric_terms_ext._area:
        errors.append("area data mismatch between generated and external grid data")

    assert not errors, "errors occured in 1x1_24:\n{}".format("\n".join(errors))


def test_extgrid_equals_generated_1x1_54():
    nx_tile, ny_tile, nz = 54, 54, 5
    comm_1by1 = MPIComm()
    cube_comm = get_cube_comm(layout=(1, 1), comm=comm_1by1)
    metric_terms_gen = MetricTerms(
        quantity_factory=get_quantity_factory(
            layout=(1, 1), nx_tile=nx_tile, ny_tile=ny_tile, nz=nz
        ),
        communicator=cube_comm,
    )

    tile_file = "/input/tilefile"
    ds = xr.open_dataset(tile_file)
    lon = ds.x.values
    lat = ds.y.values
    dx = ds.dx.values
    dy = ds.dy.values
    npx = ds.nxp.values.size
    npy = ds.nyp.values.size

    subtile_slice_grid = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_INTERFACE_DIM, pace.util.Y_INTERFACE_DIM],
        global_extent=(npx, npy),
        overlap=True,
    )

    subtile_slice_dx = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_INTERFACE_DIM, pace.util.Y_DIM],
        global_extent=(npx - 1, npy),
        overlap=True,
    )

    subtile_slice_dy = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_DIM, pace.util.Y_INTERFACE_DIM],
        global_extent=(npx, npy - 1),
        overlap=True,
    )

    metric_terms_ext = MetricTerms.from_generated(
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

    errors = []

    if metric_terms_gen.grid.data[:, :, 0] != metric_terms_ext.grid.data[:, :, 0]:
        errors.append("lon data mismatch between generated and external grid data")

    if metric_terms_gen.grid.data[:, :, 1] != metric_terms_ext.grid.data[:, :, 1]:
        errors.append("lat data mismatch between generated and external grid data")

    if metric_terms_gen._dx != metric_terms_ext._dx:
        errors.append("dx data mismatch between generated and external grid data")

    if metric_terms_gen._dy != metric_terms_ext._dy:
        errors.append("dy data mismatch between generated and external grid data")

    if metric_terms_gen._area != metric_terms_ext._area:
        errors.append("area data mismatch between generated and external grid data")

    assert not errors, "errors occured in 1x1_54:\n{}".format("\n".join(errors))


def test_extgrid_equals_generated_2x2_6():
    nx_tile, ny_tile, nz = 6, 6, 5
    comm_1by1 = MPIComm()
    cube_comm = get_cube_comm(layout=(2, 2), comm=comm_1by1)
    metric_terms_gen = MetricTerms(
        quantity_factory=get_quantity_factory(
            layout=(2, 2), nx_tile=nx_tile, ny_tile=ny_tile, nz=nz
        ),
        communicator=cube_comm,
    )

    tile_file = "/input/tilefile"
    ds = xr.open_dataset(tile_file)
    lon = ds.x.values
    lat = ds.y.values
    dx = ds.dx.values
    dy = ds.dy.values
    npx = ds.nxp.values.size
    npy = ds.nyp.values.size

    subtile_slice_grid = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_INTERFACE_DIM, pace.util.Y_INTERFACE_DIM],
        global_extent=(npx, npy),
        overlap=True,
    )

    subtile_slice_dx = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_INTERFACE_DIM, pace.util.Y_DIM],
        global_extent=(npx - 1, npy),
        overlap=True,
    )

    subtile_slice_dy = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_DIM, pace.util.Y_INTERFACE_DIM],
        global_extent=(npx, npy - 1),
        overlap=True,
    )

    metric_terms_ext = MetricTerms.from_generated(
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

    errors = []

    if metric_terms_gen.grid.data[:, :, 0] != metric_terms_ext.grid.data[:, :, 0]:
        errors.append("lon data mismatch between generated and external grid data")

    if metric_terms_gen.grid.data[:, :, 1] != metric_terms_ext.grid.data[:, :, 1]:
        errors.append("lat data mismatch between generated and external grid data")

    if metric_terms_gen._dx != metric_terms_ext._dx:
        errors.append("dx data mismatch between generated and external grid data")

    if metric_terms_gen._dy != metric_terms_ext._dy:
        errors.append("dy data mismatch between generated and external grid data")

    if metric_terms_gen._area != metric_terms_ext._area:
        errors.append("area data mismatch between generated and external grid data")

    assert not errors, "errors occured in 2x2_6:\n{}".format("\n".join(errors))


def test_extgrid_equals_generated_2x2_24():
    nx_tile, ny_tile, nz = 24, 24, 5
    comm_1by1 = MPIComm()
    cube_comm = get_cube_comm(layout=(2, 2), comm=comm_1by1)
    metric_terms_gen = MetricTerms(
        quantity_factory=get_quantity_factory(
            layout=(2, 2), nx_tile=nx_tile, ny_tile=ny_tile, nz=nz
        ),
        communicator=cube_comm,
    )

    tile_file = "/input/tilefile"
    ds = xr.open_dataset(tile_file)
    lon = ds.x.values
    lat = ds.y.values
    dx = ds.dx.values
    dy = ds.dy.values
    npx = ds.nxp.values.size
    npy = ds.nyp.values.size

    subtile_slice_grid = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_INTERFACE_DIM, pace.util.Y_INTERFACE_DIM],
        global_extent=(npx, npy),
        overlap=True,
    )

    subtile_slice_dx = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_INTERFACE_DIM, pace.util.Y_DIM],
        global_extent=(npx - 1, npy),
        overlap=True,
    )

    subtile_slice_dy = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_DIM, pace.util.Y_INTERFACE_DIM],
        global_extent=(npx, npy - 1),
        overlap=True,
    )

    metric_terms_ext = MetricTerms.from_generated(
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

    errors = []

    if metric_terms_gen.grid.data[:, :, 0] != metric_terms_ext.grid.data[:, :, 0]:
        errors.append("lon data mismatch between generated and external grid data")

    if metric_terms_gen.grid.data[:, :, 1] != metric_terms_ext.grid.data[:, :, 1]:
        errors.append("lat data mismatch between generated and external grid data")

    if metric_terms_gen._dx != metric_terms_ext._dx:
        errors.append("dx data mismatch between generated and external grid data")

    if metric_terms_gen._dy != metric_terms_ext._dy:
        errors.append("dy data mismatch between generated and external grid data")

    if metric_terms_gen._area != metric_terms_ext._area:
        errors.append("area data mismatch between generated and external grid data")

    assert not errors, "errors occured in 2x2_24:\n{}".format("\n".join(errors))


def test_extgrid_equals_generated_2x2_54():
    nx_tile, ny_tile, nz = 54, 54, 5
    comm_1by1 = MPIComm()
    cube_comm = get_cube_comm(layout=(2, 2), comm=comm_1by1)
    metric_terms_gen = MetricTerms(
        quantity_factory=get_quantity_factory(
            layout=(2, 2), nx_tile=nx_tile, ny_tile=ny_tile, nz=nz
        ),
        communicator=cube_comm,
    )

    tile_file = "/input/tilefile"
    ds = xr.open_dataset(tile_file)
    lon = ds.x.values
    lat = ds.y.values
    dx = ds.dx.values
    dy = ds.dy.values
    npx = ds.nxp.values.size
    npy = ds.nyp.values.size

    subtile_slice_grid = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_INTERFACE_DIM, pace.util.Y_INTERFACE_DIM],
        global_extent=(npx, npy),
        overlap=True,
    )

    subtile_slice_dx = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_INTERFACE_DIM, pace.util.Y_DIM],
        global_extent=(npx - 1, npy),
        overlap=True,
    )

    subtile_slice_dy = cube_comm.partitioner.subtile_slice(
        rank=cube_comm.rank,
        global_dims=[pace.util.X_DIM, pace.util.Y_INTERFACE_DIM],
        global_extent=(npx, npy - 1),
        overlap=True,
    )

    metric_terms_ext = MetricTerms.from_generated(
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

    errors = []

    if metric_terms_gen.grid.data[:, :, 0] != metric_terms_ext.grid.data[:, :, 0]:
        errors.append("lon data mismatch between generated and external grid data")

    if metric_terms_gen.grid.data[:, :, 1] != metric_terms_ext.grid.data[:, :, 1]:
        errors.append("lat data mismatch between generated and external grid data")

    if metric_terms_gen._dx != metric_terms_ext._dx:
        errors.append("dx data mismatch between generated and external grid data")

    if metric_terms_gen._dy != metric_terms_ext._dy:
        errors.append("dy data mismatch between generated and external grid data")

    if metric_terms_gen._area != metric_terms_ext._area:
        errors.append("area data mismatch between generated and external grid data")

    assert not errors, "errors occured in 2x2_54:\n{}".format("\n".join(errors))
