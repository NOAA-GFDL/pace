""" Unit tests for Jablonowski & Williamson Baroclinic test cases
Corresponds to Fortran test #12 (Steady State) and #13 (Perturbation)
found in tools/test_cases.F90 of:

https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere.git

TODO This is a place holder class for more unit tests in the future.
"""

from datetime import timedelta

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
    TileCommunicator,
    TilePartitioner,
)
from ndsl.grid import DampingCoefficients, GridData, MetricTerms
from ndsl.performance.timer import NullTimer
from pyFV3 import DycoreState, DynamicalCore, DynamicalCoreConfig


def setup_dycore_config() -> DynamicalCoreConfig:
    config = DynamicalCoreConfig(
        layout=(1, 1),
        npx=48,
        npy=48,
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
        d_con=0.0,  # Default is 0 in GFDL_atmos_cubed_sphere/model/fv_arrays.F90
        d_ext=0.0,
        dddmp=0.5,
        # delt_max=0.002,  # TODO: What is equiv in SHiELD_build?
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
        ke_bg=0.0,  # Default is 0 in GFDL_atmos_cubed_sphere/model/fv_arrays.F90
        kord_mt=9,
        kord_tm=-9,
        kord_tr=9,
        kord_wz=9,
        n_split=1,
        nord=3,
        p_fac=0.05,  # Default is 0.05 in GFDL_atmos_cubed_sphere/model/fv_arrays.F90
        rf_fast=True,
        rf_cutoff=3000.0,
        tau=10.0,
        vtdm4=0.06,
        z_tracer=True,
        do_qa=True,
        moist_phys=True,
    )
    return config


def setup_dycore(
    rank=0, usesCubedSphereComm=True, test_case=ai.AnalyticCase.baroclinic_instability
) -> DycoreState:
    """Sets up Dycore state for analytic initialization"""

    backend = "numpy"
    config = setup_dycore_config()
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
    eta_file = "tests/main/input/eta79.nc"
    metric_terms = MetricTerms(
        quantity_factory=quantity_factory,
        communicator=communicator,
        eta_file=eta_file,
    )
    grid_data = GridData.new_from_metric_terms(metric_terms)
    damping_coefficients = DampingCoefficients.new_from_metric_terms(metric_terms)

    state = ai.init_analytic_state(
        analytic_init_case=test_case,
        grid_data=grid_data,
        quantity_factory=quantity_factory,
        adiabatic=config.adiabatic,
        hydrostatic=config.hydrostatic,
        moist_phys=config.moist_phys,
        sw_dynamics=config.sw_dynamics,
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


def test_baroclinic_steady_and_instability_are_different():
    """
    Simple test to check that the steady and instability tests produce
    different results.
    """
    _, state_steady, _ = setup_dycore(test_case=ai.AnalyticCase.baroclinic_steady)
    _, state_instability, _ = setup_dycore(
        test_case=ai.AnalyticCase.baroclinic_instability
    )
    assert not (state_instability.u.field == state_steady.u.field).all()
