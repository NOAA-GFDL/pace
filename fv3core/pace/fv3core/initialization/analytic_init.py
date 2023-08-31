import math
import numpy as np
import pace.util.constants as constants
from dataclasses import fields
from types import SimpleNamespace
import pace.dsl.gt4py_utils as utils
import pace.util as fv3util
from pace.dsl.typing import Float
from pace.fv3core.initialization.dycore_state import DycoreState
from pace.util.grid import GridData, lon_lat_midpoint, great_circle_distance_lon_lat
from pace.util.grid.gnomonic import (
    get_lonlat_vect,
    get_unit_vector_direction,
    lon_lat_midpoint,
)
import pace.fv3core.initialization.init_utils as init_utils
from enum import Enum

class cases(Enum):
    baroclinic = "baroclinic"
    tropicalcylclone = "tropicalcyclone"

def init_baroclinic_state(
    grid_data: GridData,
    quantity_factory: fv3util.QuantityFactory,
    adiabatic: bool,
    hydrostatic: bool,
    moist_phys: bool,
    comm: fv3util.CubedSphereCommunicator,
) -> DycoreState:
    """
    Create a DycoreState object with quantities initialized to the Jablonowski &
    Williamson baroclinic test case perturbation applied to the cubed sphere grid.
    """
    sample_quantity = grid_data.lat
    shape = (*sample_quantity.data.shape[0:2], grid_data.ak.data.shape[0])
    nx, ny, nz = local_compute_size(shape)
    numpy_state = empty_numpy_dycore_state(shape)
    # Initializing to values the Fortran does for easy comparison
    numpy_state.delp[:] = 1e30
    numpy_state.delp[:nhalo, :nhalo] = 0.0
    numpy_state.delp[:nhalo, nhalo + ny :] = 0.0
    numpy_state.delp[nhalo + nx :, :nhalo] = 0.0
    numpy_state.delp[nhalo + nx :, nhalo + ny :] = 0.0
    numpy_state.pe[:] = 0.0
    numpy_state.pt[:] = 1.0
    numpy_state.ua[:] = 1e35
    numpy_state.va[:] = 1e35
    numpy_state.uc[:] = 1e30
    numpy_state.vc[:] = 1e30
    numpy_state.w[:] = 1.0e30
    numpy_state.delz[:] = 1.0e25
    numpy_state.phis[:] = 1.0e25
    numpy_state.ps[:] = jablo_init.surface_pressure
    eta = np.zeros(nz)
    eta_v = np.zeros(nz)
    islice, jslice, slice_3d, slice_2d = init_utils.compute_slices(nx, ny)
    # Slices with extra buffer points in the horizontal dimension
    # to accomodate averaging over shifted calculations on the grid
    _, _, slice_3d_buffer, slice_2d_buffer = init_utils.compute_slices(nx + 1, ny + 1)

    init_utils.setup_pressure_fields(
        eta=eta,
        eta_v=eta_v,
        delp=numpy_state.delp[slice_3d],
        ps=numpy_state.ps[slice_2d],
        pe=numpy_state.pe[slice_3d],
        peln=numpy_state.peln[slice_3d],
        pk=numpy_state.pk[slice_3d],
        pkz=numpy_state.pkz[slice_3d],
        ak=utils.asarray(grid_data.ak.data),
        bk=utils.asarray(grid_data.bk.data),
        ptop=grid_data.ptop,
    )

    init_utils.baroclinic_initialization(
        eta=eta,
        eta_v=eta_v,
        peln=numpy_state.peln[slice_3d_buffer],
        qvapor=numpy_state.qvapor[slice_3d_buffer],
        delp=numpy_state.delp[slice_3d_buffer],
        u=numpy_state.u[slice_3d_buffer],
        v=numpy_state.v[slice_3d_buffer],
        pt=numpy_state.pt[slice_3d_buffer],
        phis=numpy_state.phis[slice_2d_buffer],
        delz=numpy_state.delz[slice_3d_buffer],
        w=numpy_state.w[slice_3d_buffer],
        lon=utils.asarray(grid_data.lon.data[slice_2d_buffer]),
        lat=utils.asarray(grid_data.lat.data[slice_2d_buffer]),
        lon_agrid=utils.asarray(grid_data.lon_agrid.data[slice_2d_buffer]),
        lat_agrid=utils.asarray(grid_data.lat_agrid.data[slice_2d_buffer]),
        ee1=utils.asarray(grid_data.ee1.data[slice_3d_buffer]),
        ee2=utils.asarray(grid_data.ee2.data[slice_3d_buffer]),
        es1=utils.asarray(grid_data.es1.data[slice_3d_buffer]),
        ew2=utils.asarray(grid_data.ew2.data[slice_3d_buffer]),
        ptop=grid_data.ptop,
        adiabatic=adiabatic,
        hydrostatic=hydrostatic,
        nx=nx,
        ny=ny,
    )

    init_utils.p_var(
        delp=numpy_state.delp[slice_3d],
        delz=numpy_state.delz[slice_3d],
        pt=numpy_state.pt[slice_3d],
        ps=numpy_state.ps[slice_2d],
        qvapor=numpy_state.qvapor[slice_3d],
        pe=numpy_state.pe[slice_3d],
        peln=numpy_state.peln[slice_3d],
        pkz=numpy_state.pkz[slice_3d],
        ptop=grid_data.ptop,
        moist_phys=moist_phys,
        make_nh=(not hydrostatic),
    )
    state = DycoreState.init_from_numpy_arrays(
        numpy_state.__dict__,
        sizer=quantity_factory.sizer,
        backend=sample_quantity.metadata.gt4py_backend,
    )

    comm.halo_update(state.phis, n_points=nhalo)

    comm.vector_halo_update(state.u, state.v, n_points=nhalo)

    return state

def init_tc_state(
    grid_data: GridData,
    quantity_factory: fv3util.QuantityFactory,
    hydrostatic: bool,
    comm: fv3util.CubedSphereCommunicator,
) -> DycoreState:
    """
    --WARNING--WARNING--WARNING--WARNING--WARNING--WARNING--WARNING---
    --                                                              --
    --WARNING: THIS IS KNOW TO HAVE BUGS AND REQUIRE NUMERICAL DEBUG--
    --                                                              --
    --WARNING--WARNING--WARNING--WARNING--WARNING--WARNING--WARNING---
    Create a DycoreState object with quantities initialized to the
    FV3 tropical cyclone test case (test_case 55).

    This case involves a grid_transformation (done on metric terms)
    to locally increase resolution.
    """

    sample_quantity = grid_data.lat
    shape = (*sample_quantity.data.shape[:2], grid_data.ak.data.shape[0])
    numpy_state = init_utils.empty_numpy_dycore_state(shape)

    tc_properties = {
        "hydrostatic": hydrostatic,
        "dp": 1115.0,
        "exppr": 1.5,
        "exppz": 2.0,
        "gamma": 0.007,
        "lat_tc": 10.0,
        "lon_tc": 180.0,
        "p_ref": 101500.0,
        "ptop": 1.0,
        "qtrop": 1e-11,
        "q00": 0.021,
        "rp": 282000.0,
        "Ts0": 302.15,
        "vort": True,
        "ztrop": 15000.0,
        "zp": 7000.0,
        "zq1": 3000.0,
        "zq2": 8000.0,
    }

    calc = init_utils._some_inital_calculations(tc_properties)

    ps_output = init_utils._initialize_vortex_ps_phis(grid_data, shape, tc_properties, calc)
    ps, ps_u, ps_v = ps_output["ps"], ps_output["ps_uc"], ps_output["ps_vc"]

    # TODO restart file had different ak, bk. Figure out where they came from;
    # for now, take from metric terms
    ak = init_utils._define_ak()
    bk = init_utils._define_bk()
    delp = init_utils._initialize_delp(ak, bk, ps, shape)
    pe = init_utils._initialize_edge_pressure(delp, tc_properties["ptop"], shape)
    peln = np.log(pe)
    pk, pkz = init_utils.initialize_kappa_pressures(pe, peln, tc_properties["ptop"])

    pe_u = init_utils._initialize_edge_pressure_cgrid(ak, bk, ps_u, shape, tc_properties["ptop"])
    pe_v = init_utils._initialize_edge_pressure_cgrid(ak, bk, ps_v, shape, tc_properties["ptop"])

    ud, vd = init_utils._initialize_wind_dgrid(
        grid_data, tc_properties, calc, pe_u, pe_v, ps_u, ps_v, shape
    )
    ua, va = init_utils._interpolate_winds_dgrid_agrid(grid_data, ud, vd, tc_properties, shape)

    qvapor, pt = init_utils._initialize_qvapor_temperature(
        grid_data, pe, ps, tc_properties, calc, shape
    )
    delz, w = init_utils._initialize_delz_w(pe, ps, pt, qvapor, tc_properties, calc, shape)

    # numpy_state.cxd[:] =
    # numpy_state.cyd[:] =
    numpy_state.delp[:] = delp
    numpy_state.delz[:] = delz
    # numpy_state.diss_estd[:] =
    # numpy_state.mfxd[:] =
    # numpy_state.mfyd[:] =
    # numpy_state.omga[:] =
    numpy_state.pe[:] = pe
    numpy_state.peln[:] = peln
    numpy_state.phis[:] = ps_output["phis"]
    numpy_state.pk[:] = pk
    numpy_state.pkz[:] = pkz
    numpy_state.ps[:] = pe[:, :, -1]
    numpy_state.pt[:] = pt
    # numpy_state.qcld[:] =
    # numpy_state.qgraupel[:] =
    # numpy_state.qice[:] =
    # numpy_state.qliquid[:] =
    # numpy_state.qo3mr[:] =
    # numpy_state.qrain[:] =
    # numpy_state.qsgs_tke[:] =
    # numpy_state.qsnow[:] =
    numpy_state.qvapor[:] = qvapor
    # numpy_state.q_con[:] =
    numpy_state.u[:] = ud
    numpy_state.ua[:] = ua
    # numpy_state.uc[:] =
    numpy_state.v[:] = vd
    numpy_state.va[:] = va
    # numpy_state.vc[:] =
    numpy_state.w[:] = w
    breakpoint()
    state = DycoreState.init_from_numpy_arrays(
        numpy_state.__dict__,
        sizer=quantity_factory.sizer,
        backend=sample_quantity.metadata.gt4py_backend,
    )

    return state

def init_analytic_choice(
    analytic_init_str: str,
    grid_data: GridData,
    quantity_factory: fv3util.QuantityFactory,
    adiabatic: bool,
    hydrostatic: bool,
    moist_phys: bool,
    comm: fv3util.CubedSphereCommunicator,
) -> DycoreState:
    if analytic_init_str in cases:
        if analytic_init_str == "baroclinic":
            return init_baroclinic_state(
                grid_data=grid_data,
                quantity_factory=quantity_factory,
                adiabatic=False,
                hydrostatic=False,
                moist_phys=True,
                comm=comm,
                )
        
        elif analytic_init_str == "tropicalcyclone":
            return init_tc_state(
                grid_data=grid_data,
                quantity_factory=quantity_factory,
                hydrostatic=False,
                comm=comm,
            )
    else:
        raise ValueError(f"Case {analytic_init_str} not implemented")
    
    