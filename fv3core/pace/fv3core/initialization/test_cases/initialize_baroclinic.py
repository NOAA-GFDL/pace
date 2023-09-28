import math

import numpy as np

import pace.dsl.gt4py_utils as utils
import pace.fv3core.initialization.init_utils as init_utils
import pace.util as fv3util
import pace.util.constants as constants
from pace.fv3core.dycore_state import DycoreState
from pace.util.grid import GridData, great_circle_distance_lon_lat, lon_lat_midpoint


# maximum windspeed amplitude - close to windspeed of zonal-mean time-mean
# jet stream in troposphere
U0 = 35.0  # From Table VI of DCMIP2016
# [lon, lat] of zonal wind perturbation centerpoint at 20E, 40N
PCEN = [math.pi / 9.0, 2.0 * math.pi / 9.0]  # From Table VI of DCMIP2016
U1 = 1.0
SURFACE_PRESSURE = 1.0e5  # units of (Pa), from Table VI of DCMIP2016
# NOTE RADIUS = 6.3712e6 in FV3 vs Jabowski paper 6.371229e6
R = constants.RADIUS / 10.0  # Perturbation radiusfor test case 13
NHALO = fv3util.N_HALO_DEFAULT


def apply_perturbation(u_component, up, lon, lat):
    """
    Apply a Gaussian perturbation to intiate a baroclinic wave in JRMS2006
    up is the maximum amplitude of the perturbation
    modifies u_component to include the perturbation of radius R
    """
    r = np.zeros((u_component.shape[0], u_component.shape[1], 1))
    # Equation (11), distance from perturbation at 20E, 40N in JRMS2006
    r = great_circle_distance_lon_lat(PCEN[0], lon, PCEN[1], lat, constants.RADIUS, np)[
        :, :, None
    ]
    r3d = np.repeat(r, u_component.shape[2], axis=2)
    near_perturbation = (r3d / R) ** 2.0 < 40.0
    # Equation(10) in JRMS2006 perturbation applied to u_component
    # Equivalent to Equation (14) in DCMIP 2016, where Zp = 1.0
    u_component[near_perturbation] = u_component[near_perturbation] + up * np.exp(
        -((r3d[near_perturbation] / R) ** 2.0)
    )


def baroclinic_perturbed_zonal_wind(eta_v, lon, lat):
    u = zonal_wind(eta_v, lat)
    apply_perturbation(u, U1, lon, lat)
    return u


def wind_component_calc(
    shape,
    eta_v,
    lon,
    lat,
    grid_vector_component,
    islice,
    islice_grid,
    jslice,
    jslice_grid,
):
    slice_grid = (islice_grid, jslice_grid)
    slice_3d = (islice, jslice, slice(None))
    u_component = np.zeros(shape)
    u_component[slice_3d] = baroclinic_perturbed_zonal_wind(
        eta_v, lon[slice_grid], lat[slice_grid]
    )
    u_component[slice_3d] = init_utils.local_coordinate_transformation(
        u_component[slice_3d],
        lon[slice_grid],
        grid_vector_component[islice_grid, jslice_grid, :],
    )
    return u_component


def zonal_wind(eta_v, lat):
    """
    Equation (2) JRMS2006
    Returns the zonal wind u
    """
    return U0 * np.cos(eta_v[:]) ** (3.0 / 2.0) * np.sin(2.0 * lat[:, :, None]) ** 2.0


def initialize_zonal_wind(
    u,
    eta,
    eta_v,
    lon,
    lat,
    east_grid_vector_component,
    center_grid_vector_component,
    islice,
    islice_grid,
    jslice,
    jslice_grid,
    axis,
):
    shape = u.shape
    uu1 = wind_component_calc(
        shape,
        eta_v,
        lon,
        lat,
        east_grid_vector_component,
        islice,
        islice,
        jslice,
        jslice_grid,
    )
    uu3 = wind_component_calc(
        shape,
        eta_v,
        lon,
        lat,
        east_grid_vector_component,
        islice,
        islice_grid,
        jslice,
        jslice,
    )
    upper = (slice(None),) * axis + (slice(0, -1),)
    lower = (slice(None),) * axis + (slice(1, None),)
    pa1, pa2 = lon_lat_midpoint(lon[upper], lon[lower], lat[upper], lat[lower], np)
    uu2 = wind_component_calc(
        shape,
        eta_v,
        pa1,
        pa2,
        center_grid_vector_component,
        islice,
        islice,
        jslice,
        jslice,
    )
    u[islice, jslice, :] = 0.25 * (uu1 + 2.0 * uu2 + uu3)[islice, jslice, :]


def baroclinic_initialization(
    eta,
    eta_v,
    peln,
    qvapor,
    delp,
    u,
    v,
    pt,
    phis,
    delz,
    w,
    lon,
    lat,
    lon_agrid,
    lat_agrid,
    ee1,
    ee2,
    es1,
    ew2,
    ptop,
    adiabatic,
    hydrostatic,
    nx,
    ny,
):
    """
    Calls methods that compute initial state via the Jablonowski perturbation test case
    Transforms results to the cubed sphere grid
    Creates an initial baroclinic state for u(x-wind), v(y-wind), pt(temperature),
    phis(surface geopotential)w (vertical windspeed) and delz (vertical coordinate layer
    width)

    Inputs lon, lat, lon_agrid, lat_agrid, ee1, ee2, es1, ew2, ptop are defined by the
           grid and can be computed using an instance of the MetricTerms class.
    Inputs eta and eta_v are vertical coordinate columns derived from the ak and bk
           variables, also found in the Metric Terms class.
    """

    # Equation (2) for v
    # Although meridional wind is 0 in this scheme
    # on the cubed sphere grid, v is not 0 on every tile
    initialize_zonal_wind(
        v,
        eta,
        eta_v,
        lon,
        lat,
        east_grid_vector_component=ee2,
        center_grid_vector_component=ew2,
        islice=slice(0, nx + 1),
        islice_grid=slice(0, nx + 1),
        jslice=slice(0, ny),
        jslice_grid=slice(1, ny + 1),
        axis=1,
    )

    initialize_zonal_wind(
        u,
        eta,
        eta_v,
        lon,
        lat,
        east_grid_vector_component=ee1,
        center_grid_vector_component=es1,
        islice=slice(0, nx),
        islice_grid=slice(1, nx + 1),
        jslice=slice(0, ny + 1),
        jslice_grid=slice(0, ny + 1),
        axis=0,
    )

    slice_3d = (slice(0, nx), slice(0, ny), slice(None))
    slice_2d = (slice(0, nx), slice(0, ny))
    slice_2d_buffer = (slice(0, nx + 1), slice(0, ny + 1))
    # initialize temperature
    t_mean = init_utils.horizontally_averaged_temperature(eta)
    pt[slice_3d] = init_utils.cell_average_nine_components(
        init_utils.temperature,
        [eta, eta_v, t_mean],
        lon[slice_2d_buffer],
        lat[slice_2d_buffer],
        lat_agrid[slice_2d],
    )

    # initialize surface geopotential
    phis[slice_2d] = init_utils.cell_average_nine_components(
        init_utils.surface_geopotential_perturbation,
        [],
        lon[slice_2d_buffer],
        lat[slice_2d_buffer],
        lat_agrid[slice_2d],
    )

    if not hydrostatic:
        # vertical velocity is set to 0 for nonhydrostatic setups
        w[slice_3d] = 0.0
        delz[:nx, :ny, :-1] = init_utils.initialize_delz(pt[slice_3d], peln[slice_3d])

    if not adiabatic:
        qvapor[:nx, :ny, :-1] = init_utils.specific_humidity(
            delp[slice_3d], peln[slice_3d], lat_agrid[slice_2d]
        )
        pt[slice_3d] = init_utils.moisture_adjusted_temperature(
            pt[slice_3d], qvapor[slice_3d]
        )


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
    nx, ny, nz = init_utils.local_compute_size(shape)
    numpy_state = init_utils.empty_numpy_dycore_state(shape)
    # Initializing to values the Fortran does for easy comparison
    numpy_state.delp[:] = 1e30
    numpy_state.delp[:NHALO, :NHALO] = 0.0
    numpy_state.delp[:NHALO, NHALO + ny :] = 0.0
    numpy_state.delp[NHALO + nx :, :NHALO] = 0.0
    numpy_state.delp[NHALO + nx :, NHALO + ny :] = 0.0
    numpy_state.pe[:] = 0.0
    numpy_state.pt[:] = 1.0
    numpy_state.ua[:] = 1e35
    numpy_state.va[:] = 1e35
    numpy_state.uc[:] = 1e30
    numpy_state.vc[:] = 1e30
    numpy_state.w[:] = 1.0e30
    numpy_state.delz[:] = 1.0e25
    numpy_state.phis[:] = 1.0e25
    numpy_state.ps[:] = SURFACE_PRESSURE
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

    baroclinic_initialization(
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

    comm.halo_update(state.phis, n_points=NHALO)

    comm.vector_halo_update(state.u, state.v, n_points=NHALO)

    return state
