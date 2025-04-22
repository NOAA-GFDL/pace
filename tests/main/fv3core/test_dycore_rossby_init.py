""" Unit test Rossby-Haurwitz wave 4 initialization

Corresponds to Fortran shallow-water test #6 found in tools/test_cases.F90 of
https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere.git 
"""

import os
import xarray as xr
import numpy as np

from ndsl import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    NullComm,
    QuantityFactory,
    SubtileGridSizer,
    TilePartitioner,
)
from ndsl.grid import GridData, MetricTerms
import pyFV3.initialization.analytic_init as ai
from pyFV3 import DycoreState, DynamicalCoreConfig


DIR = os.path.abspath(os.path.dirname(__file__))
PACE_DIR = os.path.join(DIR, "..", "..", "..")


def setup_dycore_state(rank=0) -> DycoreState:
    """ Sets up Dycore state for Rossby analytic initialization
    """
    backend = "numpy"
    config = DynamicalCoreConfig(
        layout=(1, 1),
        npx=13,
        npy=13,
        npz=79,
        ntiles=6,
        nwat=6,
        dt_atmos=225,
        a_imp=1.0,
        beta=0.0,
        consv_te=False,  # not implemented, needs allreduce
        d2_bg=0.0,
        d2_bg_k1=0.2,
        d2_bg_k2=0.1,
        d4_bg=0.15,
        d_con=1.0,
        d_ext=0.0,
        dddmp=0.0,
        delt_max=0.002,
        do_sat_adj=True,
        do_vort_damp=True,
        fill=True,
        hord_dp=6,
        hord_mt=6,
        hord_tm=6,
        hord_tr=8,
        hord_vt=6,
        hydrostatic=False,
        k_split=1,
        ke_bg=0.0,
        kord_mt=9,
        kord_tm=-9,
        kord_tr=9,
        kord_wz=9,
        n_split=1,
        nord=3,
        p_fac=0.05,
        rf_fast=True,
        rf_cutoff=3000.0,
        tau=10.0,
        vtdm4=0.06,
        z_tracer=True,
        do_qa=True,
    )
    mpi_comm = NullComm(
        rank=rank, total_ranks=6 * config.layout[0] * config.layout[1], fill_value=0.0
    )
    partitioner = CubedSpherePartitioner(TilePartitioner(config.layout))
    communicator = CubedSphereCommunicator(mpi_comm, partitioner)
    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=config.npx - 1,
        ny_tile=config.npy - 1,
        nz=config.npz,
        n_halo=3,
        extra_dim_lengths={},
        layout=config.layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
    )
    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend=backend)
    eta_file = "tests/main/input/eta79.nc"
    metric_terms = MetricTerms(
        quantity_factory=quantity_factory,
        communicator=communicator,
        eta_file=eta_file,
    )
    grid_data = GridData.new_from_metric_terms(metric_terms)

    # create an initial state for the Rossby Wave number 4 test case
    state = ai.init_analytic_state(
        analytic_init_case="rossby",
        grid_data=grid_data,
        quantity_factory=quantity_factory,
        adiabatic=config.adiabatic,
        hydrostatic=config.hydrostatic,
        moist_phys=config.moist_phys,
        comm=communicator,
    )
    return state


def test_rossby_init():
    """ Tests Rossby-Haurwitz wave 4 initialization
        Compare initialized DycoreState values with ground truth net-cdf files.
    """
    data_dir = os.path.join(PACE_DIR, "tests", "main", "data",
                            "rossby_validation", "zero_time_restart")
    attributes = ["u", "v", "delp", "phis"]
    max_eps_error = [1e-12, 1e-12, 1e-10, 1e-14]
    for rank in range(0,6):
        fortran_rank = rank + 1
        state = setup_dycore_state(rank=rank)
        core1_ds = xr.open_dataset(os.path.join(data_dir, f"fv_core.res.tile{fortran_rank}.nc"))
        for attribute, max_eps in zip(attributes, max_eps_error):
            print(f"rank: {rank}; attribute: {attribute}")

            # Dycore values/dimensions
            state_values = getattr(state, attribute).view[:]
            state_ndims = len(getattr(state, attribute).dims)

            # Dataset values for 3D/2D Attributes at time zero
            if state_ndims == 2:  # 2D
                core1_ds_values = core1_ds[attribute].values[0, :].transpose(1, 0)
            elif state_ndims == 3: # 3D
                core1_ds_values = core1_ds[attribute].values[0, :].transpose(2, 1, 0)
            else:
                assert False, f"Unexpected number of dims in DycoreState {attribute}"

            max_error_diff = np.max(np.absolute(core1_ds_values - state_values))
            assert max_error_diff < max_eps
            # TODO: Use assert_almost_equal instead?
            np.testing.assert_almost_equal(state_values, core1_ds_values, decimal=10)

    # NOTE: The original test_cases.F90 initialized tracers for cl and cl2,
    #       but we do not initialize or check for them in this test.

    # TODO: Look at delp error epsilon further
    # TODO: Do we need sphum from fv_tracer.res.tile*.nc?
    # TODO: Do we need u_srf, v_srf from fv_srf_wnd.res.tile*.nc?
