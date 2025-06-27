""" Unit tests for Jablonowski & Williamson Baroclinic test cases 
Corresponds to Fortran test #12 (Steady State) and #13 (Perturbation) 
found in tools/test_cases.F90 of:

https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere.git
"""

import os
from datetime import timedelta
from unittest import mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from typing import Tuple


import pyFV3.initialization.analytic_init as ai
from ndsl import (
    CompilationConfig,
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    DaceConfig,
    GridIndexing,
    NullComm,
    QuantityFactory,
    StencilConfig,
    StencilFactory,
    SubtileGridSizer,
    TilePartitioner,
    TileCommunicator,
)
from ndsl.grid import DampingCoefficients, GridData, MetricTerms
from ndsl.performance.timer import NullTimer
from ndsl.constants import X_DIM, X_INTERFACE_DIM, Y_DIM, Y_INTERFACE_DIM
from ndsl.grid import (
    AngleGridData,
    ContravariantGridData,
    DampingCoefficients,
    GridData,
    HorizontalGridData,
    MetricTerms,
    VerticalGridData,
)
from ndsl.comm.partitioner import get_tile_index

from pyFV3 import DycoreState, DynamicalCore, DynamicalCoreConfig
from pace.grid import ExternalNetcdfGridConfig


DIR = os.path.abspath(os.path.dirname(__file__))
PACE_DIR = os.path.join(DIR, "..", "..", "..")
BC_DIR = os.path.join(PACE_DIR, "tests", "main", "data", "baroclinic")


@pytest.fixture()
def setenv_pace32(monkeypatch: pytest.MonkeyPatch):
    #TODO: doublecheck monkey patch fixture?
    #Monkeypatching: https://docs.pytest.org/en/stable/how-to/monkeypatch.html
    #Fixture example found here: https://stackoverflow.com/questions/77255758/how-can-i-mock-my-environment-variables-for-my-pytest
    #https://github.com/pytest-dev/pytest/issues/4576
    #https://docs.python.org/3/library/unittest.mock.html#unittest.mock.patch.dict
    with mock.patch.dict(os.environ):
        monkeypatch.setenv("PACE_FLOAT_PRECISION", "32")
        yield # Restore the environment after 


@pytest.fixture()
def setenv_pace64(monkeypatch: pytest.MonkeyPatch):
    with mock.patch.dict(os.environ):
        monkeypatch.setenv("PACE_FLOAT_PRECISION", "64")
        yield # Restore the environment after


def setup_dycore_config(test_case=ai.Cases.baroclinic) -> DynamicalCoreConfig:
    config = DynamicalCoreConfig(
        layout=(1, 1),
        npx=49,
        npy=49,
        npz=32,
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
        d_con=0.0,       # Default is 0 in GFDL_atmos_cubed_sphere/model/fv_arrays.F90
        d_ext=0.0,
        dddmp=0.5,
        #delt_max=0.002,  # TODO: What is equiv in SHiELD_build?
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
        ke_bg=0.0,       # Default is 0 in GFDL_atmos_cubed_sphere/model/fv_arrays.F90
        kord_mt=9,
        kord_tm=-9,
        kord_tr=9,
        kord_wz=9,
        n_split=1,
        nord=3,
        p_fac=0.05,      # Default is 0.05 in GFDL_atmos_cubed_sphere/model/fv_arrays.F90
        rf_fast=True,
        rf_cutoff=3000.0,
        tau=10.0,
        vtdm4=0.06,
        z_tracer=True,
        do_qa=True,
        moist_phys=True,
    )
    return config


def setup_external_grid_data_b(
        quantity_factory: QuantityFactory,
        communicator: CubedSphereCommunicator,
        eta_file: str
) -> Tuple[DampingCoefficients, GridData]:
    """TODO: flesh this out"""
    grid_file_path = os.path.join(BC_DIR, "grid_C48", "C48.BC.tile" )
    grid_type=0
    tile_num = (
        get_tile_index(communicator.rank, communicator.partitioner.total_ranks)
        + 1
    )
    tile_file = grid_file_path + str(tile_num) + ".nc"

    ds = xr.open_dataset(tile_file)
    lon = ds.x.values
    lat = ds.y.values
    npx = ds.nxp.values.size
    npy = ds.nyp.values.size

    subtile_slice_grid = communicator.partitioner.tile.subtile_slice(
        rank=communicator.rank,
        global_dims=[Y_INTERFACE_DIM, X_INTERFACE_DIM],
        global_extent=(npy, npx),
        overlap=True,
    )

    metric_terms = MetricTerms.from_external(
        x=lon[subtile_slice_grid],
        y=lat[subtile_slice_grid],
        quantity_factory=quantity_factory,
        communicator=communicator,
        grid_type=0,
        eta_file=eta_file,
    )

    horizontal_data = HorizontalGridData.new_from_metric_terms(metric_terms)
    vertical_data = VerticalGridData.new_from_metric_terms(metric_terms)
    contravariant_data = ContravariantGridData.new_from_metric_terms(metric_terms)
    angle_data = AngleGridData.new_from_metric_terms(metric_terms)
    grid_data = GridData(
        horizontal_data=horizontal_data,
        vertical_data=vertical_data,
        contravariant_data=contravariant_data,
        angle_data=angle_data,
    )

    damping_coefficients = DampingCoefficients.new_from_metric_terms(metric_terms)

    return damping_coefficients, grid_data


def setup_external_grid_data_a(
        quantity_factory: QuantityFactory,
        communicator: CubedSphereCommunicator,
        eta_file: str
) -> Tuple[DampingCoefficients, GridData]:
    """TODO: flesh this out"""
    grid_file_path = os.path.join(BC_DIR, "grid_C48", "C48.BC.tile" )
    ext_grid_config = ExternalNetcdfGridConfig(
        grid_type=0,
        grid_file_path=grid_file_path,
        eta_file=eta_file,
    )
    damping_coefficients, _, grid_data = ext_grid_config.get_grid(
        quantity_factory=quantity_factory,
        communicator=communicator,
    ) 
    return damping_coefficients, grid_data


def setup_dycore(rank=0, usesCubedSphereComm=True, test_case=ai.Cases.baroclinic.value) -> DycoreState:
    """Sets up Dycore state for analytic initialization"""
    backend = "numpy"
    config = setup_dycore_config(test_case=test_case)
    mpi_comm = NullComm(
        rank=rank, total_ranks=6 * config.layout[0] * config.layout[1], fill_value=0.0
    )
    partitioner = CubedSpherePartitioner(TilePartitioner(config.layout))

    if usesCubedSphereComm:
        communicator = CubedSphereCommunicator(mpi_comm, partitioner)
    else:
        communicator = TileCommunicator(mpi_comm, partitioner)

    dace_config = DaceConfig(communicator=communicator, backend=backend)
    stencil_config = StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend, rebuild=False, validate_args=True
        ),
        dace_config=dace_config,
    )
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
    grid_indexing = GridIndexing.from_sizer_and_communicator(
        sizer=sizer, comm=communicator
    )
    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend=backend)
    eta_file = "tests/main/input/eta32.nc" # TODO: where to document file creation for developers?
    metric_terms = MetricTerms(
        quantity_factory=quantity_factory,
        communicator=communicator,
        eta_file=eta_file,
    )
    grid_data = GridData.new_from_metric_terms(metric_terms)
    damping_coefficients = DampingCoefficients.new_from_metric_terms(metric_terms)

    #damping_coefficients2, grid_data2 = setup_external_grid_data_a(
    #    quantity_factory=quantity_factory,
    #    communicator=communicator,
    #    eta_file=eta_file,
    #)

    #damping_coefficients3, grid_data3 = setup_external_grid_data_b(
    #    quantity_factory=quantity_factory,
    #    communicator=communicator,
    #    eta_file=eta_file,
    #)

    state = ai.init_analytic_state(
        analytic_init_case=test_case,
        grid_data=grid_data,
        quantity_factory=quantity_factory,
        config=config,
        comm=communicator,
    )
    stencil_factory = StencilFactory(
        config=stencil_config,
        grid_indexing=grid_indexing,
    )

    dycore = DynamicalCore(
        comm=communicator,
        grid_data=grid_data,
        stencil_factory=stencil_factory,
        quantity_factory=quantity_factory,
        damping_coefficients=damping_coefficients,
        config=config,
        timestep=timedelta(seconds=config.dt_atmos),
        phis=state.phis,
        state=state,
    )
    return dycore, state, NullTimer()

def plot_2d_diff(testname, rank, attribute, ds_values, state_values, plot_dir="."):
    os.makedirs(plot_dir, exist_ok=True)
    diff = ds_values - state_values
    plt.title(f"diff: Fortran - Pace for '{attribute}'")
    plt.imshow(diff, cmap="viridis")
    plt.colorbar()
    plt.savefig(
        os.path.join(
            plot_dir,
            f"test_{testname}_diff_r{rank}_{attribute}.png"
        )
    )
    plt.clf()

    # Normalize the differences. Take the regular diff when ds_value is 0
    norm_diff = np.where(
        ds_values != 0, 
        np.absolute((ds_values - state_values) / ds_values), 
        np.absolute(ds_values - state_values)
    )
    plt.title(f"norm_diff: abs(Fortran - Pace / Fortran) for '{attribute}'")
    plt.imshow(norm_diff, cmap="viridis")
    plt.colorbar()
    plt.savefig(
        os.path.join(
            plot_dir,
            f"test_{testname}_norm_diff_r{rank}_{attribute}.png"
        )
    )
    plt.clf()


def plot_2d(desc, rank, attribute, data, plot_dir="."):
    os.makedirs(plot_dir, exist_ok=True)
    plt.title(f"{desc} - rank:{rank}, '{attribute}'")
    plt.imshow(data, cmap="viridis")
    plt.colorbar()
    plt.savefig(
        os.path.join(
            plot_dir,
            f"test_{desc}_r{rank}_{attribute}.png"
        )
    )
    plt.clf()


def check_init(data_dir,
               attributes,
               max_eps_errors,
               gen_plots=False,
               plot_dir=".",
               step=False,
               test_case=ai.Cases.baroclinic,
               desc="test13_64",
               rank_range=range(0,6)):
    """TODO: doc"""
    precision = "64"
    if 'PACE_FLOAT_PRECISION' in os.environ:
        precision = os.getenv("PACE_FLOAT_PRECISION", "SHOULD_NOT_BE_USED")
    print(f"precision: {precision}")
    # jk TODO: log precision instead?
    
    for rank in rank_range:
        dycore, state, timer = setup_dycore(rank=rank, test_case=test_case)

        fortran_rank = rank + 1
        if step: 
            dycore.step_dynamics(state, timer)
        core_ds = xr.open_dataset(
            os.path.join(data_dir, f"fv_core.res.tile{fortran_rank}.nc")
        )
        for attribute, max_eps_error in zip(attributes, max_eps_errors):
            print(f"rank {rank}, attribute {attribute}")
            # Dycore values/dimensions
            state_values = getattr(state, attribute.lower()).view[:]
            state_ndims = len(getattr(state, attribute.lower()).dims)

            # Dataset values for 3D/2D Attributes at time zero
            if state_ndims == 2:  # 2D
                core_ds_values = core_ds[attribute].values[0, :].transpose(1, 0)
                core_ds_values_2d, state_values_2d = core_ds_values, state_values
            elif state_ndims == 3:  # 3D
                core_ds_values = core_ds[attribute].values[0, :].transpose(2, 1, 0)
                core_ds_values_2d, state_values_2d = (
                    core_ds_values[:, :, 0],
                    state_values[:, :, 0],
                )
            else:
                assert False, f"Unexpected number of dims in DycoreState {attribute}"

            # TODO: Remove plotting eventually
            if gen_plots:
                step_prefix = "step1_" if step else "t0_"
                plot_2d(f"{step_prefix}pace.{desc}", rank, attribute, state_values_2d, plot_dir=plot_dir)
                plot_2d(f"{step_prefix}ds.{desc}", rank, attribute, core_ds_values_2d, plot_dir=plot_dir)
                plot_2d_diff(f"{step_prefix}{desc}", rank, attribute, core_ds_values_2d, state_values_2d, plot_dir=plot_dir)

            norm_diff = np.where(
                core_ds_values != 0, 
                np.absolute((core_ds_values - state_values) / core_ds_values), 
                np.absolute(core_ds_values - state_values)
            )
            max_error_norm_diff = np.max(norm_diff) # TODO: Use this when values don't blow up...?
            max_error_diff = np.max(np.abs(core_ds_values - state_values))
            assert max_error_diff < max_eps_error

    # NOTE: The original test_cases.F90 initialized tracers for cl and cl2,
    #       but we do not initialize or check for them in this test.



# TODO: Why doesn't this work instead of setenv_pace64?
#@mock.patch.dict("os.environ", {"PACE_FLOAT_PRECISION", "64"})
#def test_rossby_init64():
def test_baroclinic_init64(setenv_pace64: None):
    """Tests case #13 (Perturbation) initialization for 64bit precision
    Compare initialized DycoreState values with ground truth net-cdf files.

    Ground truth RESTART files were generated using the following script: 
    TODO (where to put the script?)
    Using SHiELD_build 8309e1151812f72dda41142a16da0eb1f2bc4f8a (5/22/2025)
    """
    desc = "test13_64_debug_rs"
    data_dir = os.path.join(BC_DIR, desc)
    attributes = ["phis", "delp", "u", "v"] # TODO: more attributes
    max_eps_errors = [5e-12, 1e-14, 2e-12, 2e-12]
    check_init(data_dir, attributes, max_eps_errors, test_case=ai.Cases.baroclinic.value, gen_plots=True, desc=desc, plot_dir=desc)


def test_baroclinic_init32(setenv_pace32: None):
    """Tests case #13 (Perturbation) initialization for 32bit precision
    Compare initialized DycoreState values with ground truth net-cdf files.
    """
    desc = "test13_32_debug_rs"
    data_dir = os.path.join(BC_DIR, desc)
    attributes = ["phis", "delp", "u", "v"] # TODO: more attributes
    max_eps_errors = [8e-4, 4e-3, 8e-6, 8e-6]
    check_init(data_dir, attributes, max_eps_errors, test_case=ai.Cases.baroclinic.value, gen_plots=True, desc=desc, plot_dir=desc)


def test_baroclinic_12_init64(setenv_pace64: None):
    """Tests case #12 (Steady State) initialization for 64bit precision
    Compare initialized DycoreState values with ground truth net-cdf files.
    """
    desc = "test12_64_debug_rs"
    data_dir = os.path.join(BC_DIR, desc)
    attributes = ["phis", "delp", "u", "v"] # TODO: more attributes
    max_eps_errors = [5e-12, 1e-14, 2e-12, 2e-12]
    check_init(data_dir, attributes, max_eps_errors, test_case=ai.Cases.baroclinic_ss.value, gen_plots=True, desc=desc, plot_dir=desc)


def test_baroclinic_12_init32(setenv_pace32: None):
    """Tests case #12 (Steady State) initialization for 32bit precision
    Compare initialized DycoreState values with ground truth net-cdf files.
    """
    #data_dir = os.path.join(BC_DIR, "C96.solo.BCmoist.pace_12_32") # TODO: REMOVE
    desc = "test12_32_debug_rs"
    data_dir = os.path.join(BC_DIR, desc)
    attributes = ["phis", "delp", "u", "v"] # TODO: more attributes
    max_eps_errors = [8e-4, 4e-3, 8e-6, 8e-6]
    check_init(data_dir, attributes, max_eps_errors, test_case=ai.Cases.baroclinic_ss.value, gen_plots=True, desc=desc, plot_dir=desc)


def test_tmp_grid_checks(setenv_pace64: None):
    """
    Compare SHiELD restart and Pace output grid net-cdf files.
    """
    # Load Pace grid data - x, y (requires conversion from radians to ??? TODO)
    pace_out_dir = "/home/Janice.Kim/pace/baroclinic_comparison_20250613_a/output.netcfg.baroclinic_c48_test13"
    pace_lat_path = os.path.join(pace_out_dir, "constants_lat.nc")
    pace_lat_ds = xr.open_dataset(pace_lat_path)

    # Load SHiELD grid data - from Fre NC Tools make_hgrid
    backend = "numpy"
    config = setup_dycore_config(test_case="baroclinic_ss")
    mpi_comm = NullComm(
        rank=0, total_ranks=6 * config.layout[0] * config.layout[1], fill_value=0.0
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
    eta_file = "tests/main/input/eta32.nc"

    _, grid_data = setup_external_grid_data_a(
        quantity_factory=quantity_factory,
        communicator=communicator,
        eta_file=eta_file,
    )
    shield_lat = grid_data.lat.field

    # Compare
    # shield_lat[:] should be == pace_lat_ds.lat[0].data[:]
    # Sigh... but what does this prove? They should be the same.
    # I think I need to go the other way... convert pace to cartesian and see if it matches
    # the FRE NC tools generated tile files. (?) TODO

    pass


# TODO: jk delete this before merge; temp test
def test_rusty(setenv_pace64: None): 
    shield_path = '/home/Rusty.Benson/SHiELD/SHiELD_build/CI/BATCH-CI/C48.solo.BCmoist/RESTART/fv_core.res.tile1.nc'
    pace_path = '/home/Rusty.Benson/PACE/pace/RESTART/restart_dycore_state_0.nc'

    attribute = 'u'

    shield_ds = xr.open_dataset(shield_path)
    time = 0
    z = 0
    shield_xy0 = shield_ds['u'][time][z].T

    num_halos = 3
    pace_ds = xr.open_dataset(pace_path)
    # TODO: I know that the x isn't x_interface, I should be able to figure that out without manually compensating.
    pace_xy0 = pace_ds['u'][num_halos:-(num_halos+1),num_halos:-num_halos,z]

    rank = 0
    plot_dir = 'compare_rusty'
    plot_2d(f"shield", rank, attribute, shield_xy0, plot_dir=plot_dir)
    plot_2d(f"pace", rank, attribute, pace_xy0, plot_dir=plot_dir)


    os.makedirs(plot_dir, exist_ok=True)
    diff = shield_xy0.data - pace_xy0.data
    plt.title(f"diff: SHiELD - Pace for '{attribute}'")
    plt.imshow(diff, cmap="viridis")
    plt.colorbar()
    plt.savefig(
        os.path.join(
            plot_dir,
            f"test_diff_r{rank}_{attribute}.png"
        )
    )
    plt.clf()

    #not_zero_mask = diff[diff != 0] # TODO: jk zero comparison okay?
    #norm_diff = diff.copy()
    #norm_diff[not_zero_mask] = np.absolute((ds_values - state_values) / ds_values)
    norm_diff = np.absolute((shield_xy0.data - pace_xy0.data) / shield_xy0.data)
    plt.title(f"norm_diff: abs(Fortran - Pace / Fortran) for '{attribute}'")
    plt.imshow(norm_diff, cmap="viridis")
    plt.colorbar()
    plt.savefig(
        os.path.join(
            plot_dir,
            f"test_norm_diff_r{rank}_{attribute}.png"
        )
    )
    plt.clf()

    max_eps_error = 1e-12

    shield_minus_pace = shield_xy0.data - pace_xy0.data
    max_error_diff = np.max(
        np.absolute((shield_minus_pace) / shield_xy0.data)
    )
    if np.isnan(max_error_diff): # e.g., from zero division
        max_error_diff = np.max(np.absolute(shield_minus_pace))

    assert max_error_diff < max_eps_error
