import os

import numpy as np
import xarray as xr

from ndsl import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    LocalComm,
    NullComm,
    QuantityFactory,
    SubtileGridSizer,
    TilePartitioner,
)
from pace.driver import FortranRestartInit, GeneratedGridConfig
from pySHiELD import PHYSICS_PACKAGES


DIR = os.path.dirname(os.path.abspath(__file__))
PACE_DIR = os.path.join(DIR, "../../../")


def test_state_from_fortran_restart():
    layout = (1, 1)
    partitioner = CubedSpherePartitioner(TilePartitioner(layout))
    # need a local communicator to mock "scatter" for the restart data,
    # but need null communicator to handle grid initialization
    local_comm = LocalComm(rank=0, total_ranks=6, buffer_dict={})
    null_comm = NullComm(rank=0, total_ranks=6)
    local_communicator = CubedSphereCommunicator(local_comm, partitioner)
    null_communicator = CubedSphereCommunicator(null_comm, partitioner)

    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=12,
        ny_tile=12,
        nz=63,
        n_halo=3,
        extra_dim_lengths={},
        layout=layout,
        tile_partitioner=partitioner.tile,
        tile_rank=0,
    )

    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend="numpy")
    restart_dir = os.path.join(PACE_DIR, "tests/main/data/c12_restart")

    (damping_coefficients, driver_grid_data, grid_data,) = GeneratedGridConfig(
        restart_path=restart_dir, eta_file=restart_dir + "/fv_core.res.nc"
    ).get_grid(quantity_factory, null_communicator)

    restart_config = FortranRestartInit(path=restart_dir)
    driver_state = restart_config.get_driver_state(
        quantity_factory,
        local_communicator,
        damping_coefficients=damping_coefficients,
        driver_grid_data=driver_grid_data,
        grid_data=grid_data,
        schemes=[PHYSICS_PACKAGES["GFS_microphysics"]],
    )
    ds = xr.open_dataset(os.path.join(restart_dir, "fv_core.res.tile1.nc"))
    np.testing.assert_array_equal(
        ds["u"].values[0, :].transpose(2, 1, 0), driver_state.dycore_state.u.view[:]
    )
