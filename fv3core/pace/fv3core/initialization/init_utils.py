import math
from dataclasses import fields
from types import SimpleNamespace

import numpy as np

import pace.util as fv3util
import pace.util.constants as constants
from pace.dsl.typing import Float
from pace.fv3core.dycore_state import DycoreState
from pace.util.grid import lon_lat_midpoint
from pace.util.grid.gnomonic import get_lonlat_vect, get_unit_vector_direction


# maximum windspeed amplitude - close to windspeed of zonal-mean time-mean
# jet stream in troposphere
U0 = 35.0  # From Table VI of DCMIP2016
# [lon, lat] of zonal wind perturbation centerpoint at 20E, 40N
PCEN = [math.pi / 9.0, 2.0 * math.pi / 9.0]  # From Table VI of DCMIP2016
PTOP_MIN = 1e-8
U1 = 1.0
PT0 = 0.0
ETA_0 = 0.252
ETA_SURFACE = 1.0
ETA_TROPOPAUSE = 0.2
T_0 = 288.0
DELTA_T = 480000.0
LAPSE_RATE = 0.005  # From Table VI of DCMIP2016
SURFACE_PRESSURE = 1.0e5  # units of (Pa), from Table VI of DCMIP2016
# NOTE RADIUS = 6.3712e6 in FV3 vs Jabowski paper 6.371229e6
R = constants.RADIUS / 10.0  # Perturbation radiusfor test case 13
NHALO = fv3util.N_HALO_DEFAULT


def cell_average_nine_components(
    component_function,
    component_args,
    lon,
    lat,
    lat_agrid,
):
    """
    Outputs the weighted average of a field that is a function of latitude,
    averaging over the 9 points on the corners, edges, and center of each
    gridcell.

    Args:
        component_function: callable taking in an array of latitude and
            returning an output array
        component_args: arguments to pass on to component_function,
            should not be a function of latitude
        lon: longitude array, defined on cell corners
        lat: latitude array, defined on cell corners
        lat_agrid: latitude array, defined on cell centers
    """
    # this weighting is done to reproduce the behavior of the Fortran code
    # Compute cell lats in the midpoint of each cell edge
    lat2, lat3, lat4, lat5 = compute_grid_edge_midpoint_latitude_components(lon, lat)
    pt1 = component_function(*component_args, lat=lat_agrid)
    pt2 = component_function(*component_args, lat=lat2[:, :-1])
    pt3 = component_function(*component_args, lat=lat3)
    pt4 = component_function(*component_args, lat=lat4)
    pt5 = component_function(*component_args, lat=lat5[:-1, :])
    pt6 = component_function(*component_args, lat=lat[:-1, :-1])
    pt7 = component_function(*component_args, lat=lat[1:, :-1])
    pt8 = component_function(*component_args, lat=lat[1:, 1:])
    pt9 = component_function(*component_args, lat=lat[:-1, 1:])
    return cell_average_nine_point(pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9)


def cell_average_nine_point(pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9):
    """
    9-point average: should be 2nd order accurate for a rectangular cell
    9  4  8
    5  1  3
    6  2  7
    """
    return (
        0.25 * pt1 + 0.125 * (pt2 + pt3 + pt4 + pt5) + 0.0625 * (pt6 + pt7 + pt8 + pt9)
    )


# TODO: Many duplicate functions do this exact calculation, we should consolidate them
def compute_eta(ak, bk):
    """
    Equation (1) JRMS2006
    eta is the vertical coordinate and eta_v is an auxiliary vertical coordinate
    """
    eta = 0.5 * ((ak[:-1] + ak[1:]) / SURFACE_PRESSURE + bk[:-1] + bk[1:])
    eta_v = vertical_coordinate(eta)
    return eta, eta_v


def compute_grid_edge_midpoint_latitude_components(lon, lat):
    _, lat_avg_x_south = lon_lat_midpoint(
        lon[0:-1, :], lon[1:, :], lat[0:-1, :], lat[1:, :], np
    )
    _, lat_avg_y_east = lon_lat_midpoint(
        lon[1:, 0:-1], lon[1:, 1:], lat[1:, 0:-1], lat[1:, 1:], np
    )
    _, lat_avg_x_north = lon_lat_midpoint(
        lon[0:-1, 1:], lon[1:, 1:], lat[0:-1, 1:], lat[1:, 1:], np
    )
    _, lat_avg_y_west = lon_lat_midpoint(
        lon[:, 0:-1], lon[:, 1:], lat[:, 0:-1], lat[:, 1:], np
    )
    return lat_avg_x_south, lat_avg_y_east, lat_avg_x_north, lat_avg_y_west


def compute_slices(nx, ny):
    islice = slice(NHALO, NHALO + nx)
    jslice = slice(NHALO, NHALO + ny)
    slice_3d = (islice, jslice, slice(None))
    slice_2d = (islice, jslice)
    return islice, jslice, slice_3d, slice_2d


def empty_numpy_dycore_state(shape):
    numpy_dict = {}
    for _field in fields(DycoreState):
        if "dims" in _field.metadata.keys():
            numpy_dict[_field.name] = np.zeros(
                shape[: len(_field.metadata["dims"])],
                dtype=Float,
            )
    numpy_state = SimpleNamespace(**numpy_dict)
    return numpy_state


def _find_midpoint_unit_vectors(p1, p2):

    midpoint = np.array(
        lon_lat_midpoint(p1[:, :, 0], p2[:, :, 0], p1[:, :, 1], p2[:, :, 1], np)
    ).transpose([1, 2, 0])
    unit_dir = get_unit_vector_direction(p1, p2, np)
    exv, eyv = get_lonlat_vect(midpoint, np)

    muv = {"midpoint": midpoint, "unit_dir": unit_dir, "exv": exv, "eyv": eyv}

    return muv


def fix_top_log_edge_pressure(peln, ptop):
    if ptop < PTOP_MIN:
        ak1 = (constants.KAPPA + 1.0) / constants.KAPPA
        peln[:, :, 0] = peln[:, :, 1] - ak1
    else:
        peln[:, :, 0] = np.log(ptop)


def geopotential_perturbation(lat, eta_value):
    """
    Equation (7) JRMS2006, just the perturbation component
    """
    u_comp = U0 * (np.cos(eta_value) ** (3.0 / 2.0))
    return u_comp * (
        (-2.0 * (np.sin(lat) ** 6.0) * (np.cos(lat) ** 2.0 + 1.0 / 3.0) + 10.0 / 63.0)
        * u_comp
        + (
            (8.0 / 5.0) * (np.cos(lat) ** 3.0) * (np.sin(lat) ** 2.0 + 2.0 / 3.0)
            - math.pi / 4.0
        )
        * constants.RADIUS
        * constants.OMEGA
    )


def horizontally_averaged_temperature(eta):
    """
    Equations (4) and (5) JRMS2006 for characteristic temperature profile
    """
    # for troposphere:
    t_mean = T_0 * eta[:] ** (constants.RDGAS * LAPSE_RATE / constants.GRAV)
    # above troposphere
    t_mean[ETA_TROPOPAUSE > eta] = (
        t_mean[ETA_TROPOPAUSE > eta]
        + DELTA_T * (ETA_TROPOPAUSE - eta[ETA_TROPOPAUSE > eta]) ** 5.0
    )
    return t_mean


def initialize_delp(ps, ak, bk):
    return (
        ak[None, None, 1:]
        - ak[None, None, :-1]
        + ps[:, :, None] * (bk[None, None, 1:] - bk[None, None, :-1])
    )


def initialize_delz(pt, peln):
    return constants.RDG * pt[:, :, :-1] * (peln[:, :, 1:] - peln[:, :, :-1])


def initialize_edge_pressure(delp, ptop):
    pe = np.zeros(delp.shape)
    pe[:, :, 0] = ptop
    for k in range(1, pe.shape[2]):
        pe[:, :, k] = pe[:, :, k - 1] + delp[:, :, k - 1]
    return pe


def _initialize_edge_pressure_cgrid(ak, bk, ps, shape, ptop):
    """
    Initialize edge pressure on c-grid for u and v points,
    depending on which ps is input (ps_uc or ps_vc)
    """
    pe_cgrid = np.zeros(shape)
    pe_cgrid[:, :, 0] = ptop

    pe_cgrid[:, :, :] = ak[None, None, :] + ps[:, :, None] * bk[None, None, :]

    return pe_cgrid


def initialize_kappa_pressures(pe, peln, ptop):
    """
    Compute the edge_pressure**kappa (pk) and the layer mean of this (pkz)
    """
    pk = np.zeros(pe.shape)
    pkz = np.zeros(pe.shape)
    pk[:, :, 0] = ptop ** constants.KAPPA
    pk[:, :, 1:] = np.exp(constants.KAPPA * np.log(pe[:, :, 1:]))
    pkz[:, :, :-1] = (pk[:, :, 1:] - pk[:, :, :-1]) / (
        constants.KAPPA * (peln[:, :, 1:] - peln[:, :, :-1])
    )
    return pk, pkz


def initialize_log_pressure_interfaces(pe, ptop):
    peln = np.zeros(pe.shape)
    peln[:, :, 0] = math.log(ptop)
    peln[:, :, 1:] = np.log(pe[:, :, 1:])
    return peln


def initialize_pkz_dry(delp, pt, delz):
    return np.exp(
        constants.KAPPA
        * np.log(constants.RDG * delp[:, :, :-1] * pt[:, :, :-1] / delz[:, :, :-1])
    )


def initialize_pkz_moist(delp, pt, qvapor, delz):
    return np.exp(
        constants.KAPPA
        * np.log(
            constants.RDG
            * delp[:, :, :-1]
            * pt[:, :, :-1]
            * (1.0 + constants.ZVIR * qvapor[:, :, :-1])
            / delz[:, :, :-1]
        )
    )


def local_compute_size(data_array_shape):
    nx = data_array_shape[0] - 2 * NHALO - 1
    ny = data_array_shape[1] - 2 * NHALO - 1
    nz = data_array_shape[2]
    return nx, ny, nz


def local_coordinate_transformation(u_component, lon, grid_vector_component):
    """
    Transform the zonal wind component to the cubed sphere grid using a grid vector
    """
    return (
        u_component
        * (
            grid_vector_component[:, :, 1] * np.cos(lon)
            - grid_vector_component[:, :, 0] * np.sin(lon)
        )[:, :, None]
    )


def moisture_adjusted_temperature(pt, qvapor):
    """
    Update initial temperature to include water vapor contribution
    """
    return pt / (1.0 + constants.ZVIR * qvapor)


def p_var(
    delp,
    delz,
    pt,
    ps,
    qvapor,
    pe,
    peln,
    pkz,
    ptop,
    moist_phys,
    make_nh,
):
    """
    Computes auxiliary pressure variables for a hydrostatic state.

    The Fortran code also recomputes some more pressure variables,
    pe, pk, but since these are already done in setup_pressure_fields
    we don't duplicate them here
    """

    ps[:] = pe[:, :, -1]
    fix_top_log_edge_pressure(peln, ptop)

    if make_nh:
        delz[:, :, :-1] = initialize_delz(pt, peln)
    if moist_phys:
        pkz[:, :, :-1] = initialize_pkz_moist(delp, pt, qvapor, delz)
    else:
        pkz[:, :, :-1] = initialize_pkz_dry(delp, pt, delz)


def setup_pressure_fields(
    eta,
    eta_v,
    delp,
    ps,
    pe,
    peln,
    pk,
    pkz,
    ak,
    bk,
    ptop,
):
    ps[:] = SURFACE_PRESSURE
    delp[:, :, :-1] = initialize_delp(ps, ak, bk)
    pe[:] = initialize_edge_pressure(delp, ptop)
    peln[:] = initialize_log_pressure_interfaces(pe, ptop)
    pk[:], pkz[:] = initialize_kappa_pressures(pe, peln, ptop)
    eta[:-1], eta_v[:-1] = compute_eta(ak, bk)


def specific_humidity(delp, peln, lat_agrid):
    """
    Compute specific humidity using the DCMPI2016 equation 18 and relevant constants
    """
    #  Specific humidity vertical pressure width parameter (Pa)
    pw = 34000.0
    # Maximum specific humidity amplitude (kg/kg) for Idealized Tropical Cyclone test
    # TODO: should we be using 0.018, the baroclinic wave test instead?
    q0 = 0.021
    # In equation 18 of DCMPI2016, ptmp is pressure - surface pressure
    # TODO why do we use dp/(d(log(p))) for 'pressure'?
    ptmp = delp[:, :, :-1] / (peln[:, :, 1:] - peln[:, :, :-1]) - SURFACE_PRESSURE
    # Similar to equation 18 of DCMIP2016 without a cutoff at tropopause
    return (
        q0
        * np.exp(-((lat_agrid[:, :, None] / PCEN[1]) ** 4.0))
        * np.exp(-((ptmp / pw) ** 2.0))
    )


def surface_geopotential_perturbation(lat):
    """
    From JRMS2006:
    * 'In hydrostatic models with pressure-based vertical coordinates, it's
       only necessary to initialize surface geopotential.'
    * 'balances the non-zero zonal wind at the surface with surface elevation zs'
    """
    surface_level = vertical_coordinate(ETA_SURFACE)
    return geopotential_perturbation(lat, surface_level)


def temperature(eta, eta_v, t_mean, lat):
    """
    Equation (6)JRMS2006
    The total temperature distribution from the horizontal-mean temperature
     and a horizontal variation at each level
    """
    lat = lat[:, :, None]
    return t_mean + 0.75 * (eta[:] * math.pi * U0 / constants.RDGAS) * np.sin(
        eta_v[:]
    ) * np.sqrt(np.cos(eta_v[:])) * (
        (-2.0 * (np.sin(lat) ** 6.0) * (np.cos(lat) ** 2.0 + 1.0 / 3.0) + 10.0 / 63.0)
        * 2.0
        * U0
        * np.cos(eta_v[:]) ** (3.0 / 2.0)
        + (
            (8.0 / 5.0) * (np.cos(lat) ** 3.0) * (np.sin(lat) ** 2.0 + 2.0 / 3.0)
            - math.pi / 4.0
        )
        * constants.RADIUS
        * constants.OMEGA
    )


def vertical_coordinate(eta_value):
    """
    Equation (1) JRMS2006
    computes eta_v, the auxiliary variable vertical coordinate
    """
    return (eta_value - ETA_0) * math.pi * 0.5
