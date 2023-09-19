import math
from dataclasses import fields
from types import SimpleNamespace

import numpy as np

import pace.util as fv3util
import pace.util.constants as constants
from pace.dsl.typing import Float

# from pace.fv3core.initialization.dycore_state import DycoreState
from pace.fv3core.dycore_state import DycoreState
from pace.util.grid import great_circle_distance_lon_lat, lon_lat_midpoint
from pace.util.grid.gnomonic import get_lonlat_vect, get_unit_vector_direction


# maximum windspeed amplitude - close to windspeed of zonal-mean time-mean
# jet stream in troposphere
u0 = 35.0  # From Table VI of DCMIP2016
# [lon, lat] of zonal wind perturbation centerpoint at 20E, 40N
pcen = [math.pi / 9.0, 2.0 * math.pi / 9.0]  # From Table VI of DCMIP2016
ptop_min = 1e-8
u1 = 1.0
pt0 = 0.0
eta_0 = 0.252
eta_surface = 1.0
eta_tropopause = 0.2
t_0 = 288.0
delta_t = 480000.0
lapse_rate = 0.005  # From Table VI of DCMIP2016
surface_pressure = 1.0e5  # units of (Pa), from Table VI of DCMIP2016
# NOTE RADIUS = 6.3712e6 in FV3 vs Jabowski paper 6.371229e6
R = constants.RADIUS / 10.0  # Perturbation radiusfor test case 13
nhalo = fv3util.N_HALO_DEFAULT


def apply_perturbation(u_component, up, lon, lat):
    """
    Apply a Gaussian perturbation to intiate a baroclinic wave in JRMS2006
    up is the maximum amplitude of the perturbation
    modifies u_component to include the perturbation of radius R
    """
    r = np.zeros((u_component.shape[0], u_component.shape[1], 1))
    # Equation (11), distance from perturbation at 20E, 40N in JRMS2006
    r = great_circle_distance_lon_lat(pcen[0], lon, pcen[1], lat, constants.RADIUS, np)[
        :, :, None
    ]
    r3d = np.repeat(r, u_component.shape[2], axis=2)
    near_perturbation = (r3d / R) ** 2.0 < 40.0
    # Equation(10) in JRMS2006 perturbation applied to u_component
    # Equivalent to Equation (14) in DCMIP 2016, where Zp = 1.0
    u_component[near_perturbation] = u_component[near_perturbation] + up * np.exp(
        -((r3d[near_perturbation] / R) ** 2.0)
    )


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
    t_mean = horizontally_averaged_temperature(eta)
    pt[slice_3d] = cell_average_nine_components(
        temperature,
        [eta, eta_v, t_mean],
        lon[slice_2d_buffer],
        lat[slice_2d_buffer],
        lat_agrid[slice_2d],
    )

    # initialize surface geopotential
    phis[slice_2d] = cell_average_nine_components(
        surface_geopotential_perturbation,
        [],
        lon[slice_2d_buffer],
        lat[slice_2d_buffer],
        lat_agrid[slice_2d],
    )

    if not hydrostatic:
        # vertical velocity is set to 0 for nonhydrostatic setups
        w[slice_3d] = 0.0
        delz[:nx, :ny, :-1] = initialize_delz(pt[slice_3d], peln[slice_3d])

    if not adiabatic:
        qvapor[:nx, :ny, :-1] = specific_humidity(
            delp[slice_3d], peln[slice_3d], lat_agrid[slice_2d]
        )
        pt[slice_3d] = moisture_adjusted_temperature(pt[slice_3d], qvapor[slice_3d])


def baroclinic_perturbed_zonal_wind(eta_v, lon, lat):
    u = zonal_wind(eta_v, lat)
    apply_perturbation(u, u1, lon, lat)
    return u


def _calculate_distance_from_tc_center(pe_v, ps_v, muv, calc, tc_properties):

    d1 = np.sin(calc["p0"][1]) * np.cos(muv["midpoint"][:, :, 1]) - np.cos(
        calc["p0"][1]
    ) * np.sin(muv["midpoint"][:, :, 1]) * np.cos(
        muv["midpoint"][:, :, 0] - calc["p0"][0]
    )
    d2 = np.cos(calc["p0"][1]) * np.sin(muv["midpoint"][:, :, 0] - calc["p0"][0])
    d = np.sqrt(d1 ** 2 + d2 ** 2)
    d[d < 1e-15] = 1e-15

    r = great_circle_distance_lon_lat(
        calc["p0"][0],
        muv["midpoint"][:, :, 0],
        calc["p0"][1],
        muv["midpoint"][:, :, 1],
        constants.RADIUS,
        np,
    )
    ptmp = 0.5 * (pe_v[:, :, :-1] + pe_v[:, :, 1:])
    height = (calc["t00"] / tc_properties["gamma"]) * (
        1.0 - (ptmp / ps_v[:, :, None]) ** calc["exponent"]
    )

    distance_dict = {"d": d, "d1": d1, "d2": d2, "height": height, "r": r}

    return distance_dict


def _calculate_pt_height(height, qvapor, r, tc_properties, calc):

    aa = height / tc_properties["zp"]
    bb = np.exp(aa ** tc_properties["exppz"])
    cc = r / tc_properties["rp"]
    dd = np.exp(cc ** tc_properties["exppr"])
    ee = 1.0 - tc_properties["p_ref"] / tc_properties["dp"] * dd[:, :, None] * bb
    ff = constants.GRAV * tc_properties["zp"] ** tc_properties["exppz"] * ee
    gg = calc["t00"] - tc_properties["gamma"] * height
    hh = 1.0 + tc_properties["exppz"] * constants.RDGAS * gg * height / ff
    ii = 1.0 + constants.ZVIR * qvapor

    pt = gg / ii / hh

    return pt


def _calculate_utmp(height, dist, calc, tc_properties):

    aa = height / tc_properties["zp"]  # (134, 135, 79)
    bb = dist["r"] / tc_properties["rp"]  # (134, 135)
    cc = aa ** tc_properties["exppz"]  # (134, 135, 79)
    dd = bb ** tc_properties["exppr"]  # (134, 135)
    ee = 1.0 - tc_properties["p_ref"] / tc_properties["dp"] * np.exp(
        dd[:, :, None]
    ) * np.exp(
        cc
    )  # (134, 135, 79)
    ff = constants.GRAV * tc_properties["zp"] ** tc_properties["exppz"]  # number
    gg = calc["t00"] - tc_properties["gamma"] * height  # (134, 135, 79)
    hh = (
        tc_properties["exppz"] * height * constants.RDGAS * gg / ff + ee
    )  # (134, 135, 79)
    ii = calc["cor"] * dist["r"] / 2.0  # (134, 135)
    jj = 2.0  # number
    kk = (
        ii[:, :, None] ** jj
        - tc_properties["exppr"]
        * bb[:, :, None] ** tc_properties["exppr"]
        * constants.RDGAS
        * gg
        / hh
    )  # (134, 135, 79)
    ll = -calc["cor"] * dist["r"][:, :, None] / 2.0 + np.sqrt(kk)  # (134, 135, 79)

    utmp = 1.0 / dist["d"][:, :, None] * ll

    return utmp


def _calculate_vortex_surface_pressure_with_radius(p0, p_grid, tc_properties):
    """
    p0 is the tc center point
    p_grid is the grid_data.grid variable corresponding to what is needed
    for ps on A-grid, p_grid is grid_data.agrid.data
    """

    r = great_circle_distance_lon_lat(
        p0[0], p_grid[:, :, 0], p0[1], p_grid[:, :, 1], constants.RADIUS, np
    )
    ps = tc_properties["p_ref"] - tc_properties["dp"] * np.exp(
        -((r / tc_properties["rp"]) ** 1.5)
    )

    return ps


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


def compute_eta(ak, bk):
    """
    Equation (1) JRMS2006
    eta is the vertical coordinate and eta_v is an auxiliary vertical coordinate
    """
    eta = 0.5 * ((ak[:-1] + ak[1:]) / surface_pressure + bk[:-1] + bk[1:])
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
    islice = slice(nhalo, nhalo + nx)
    jslice = slice(nhalo, nhalo + ny)
    slice_3d = (islice, jslice, slice(None))
    slice_2d = (islice, jslice)
    return islice, jslice, slice_3d, slice_2d


def _define_ak():
    ak = np.array(
        [
            1.0,
            4.3334675,
            10.677089,
            21.541527,
            35.58495,
            52.180374,
            71.43384,
            93.80036,
            120.00996,
            151.22435,
            188.9106,
            234.46414,
            289.17712,
            354.48535,
            431.97327,
            523.37726,
            630.5866,
            755.64453,
            900.745,
            1068.2289,
            1260.578,
            1480.4054,
            1730.4465,
            2013.547,
            2332.6455,
            2690.7593,
            3090.968,
            3536.3882,
            4030.164,
            4575.4316,
            5175.3115,
            5832.871,
            6551.1157,
            7332.957,
            8181.1987,
            9098.508,
            10087.395,
            11024.773,
            11800.666,
            12434.538,
            12942.799,
            13339.364,
            13636.07,
            13843.023,
            13968.898,
            14021.17,
            14006.316,
            13929.978,
            13797.103,
            13612.045,
            13378.674,
            13100.442,
            12780.452,
            12421.524,
            12026.23,
            11596.926,
            11135.791,
            10644.859,
            10126.039,
            9581.128,
            9011.82,
            8419.746,
            7806.45,
            7171.0674,
            6517.474,
            5861.0713,
            5214.543,
            4583.264,
            3972.567,
            3387.6143,
            2833.4417,
            2314.8213,
            1836.298,
            1402.0687,
            1016.004,
            681.5313,
            401.6795,
            178.95885,
            0.0015832484,
            0.0,
        ]
    )

    return ak


def _define_bk():
    bk = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0012543945,
            0.004884307,
            0.01071458,
            0.018598301,
            0.028411143,
            0.040047247,
            0.05341539,
            0.068436176,
            0.08503932,
            0.10316211,
            0.12274763,
            0.14374262,
            0.16609776,
            0.18976559,
            0.21470062,
            0.24085836,
            0.2681944,
            0.2966648,
            0.32622582,
            0.35683352,
            0.38844362,
            0.421011,
            0.45448995,
            0.4888354,
            0.52400005,
            0.55993766,
            0.59673643,
            0.63420564,
            0.67150474,
            0.7079659,
            0.74333787,
            0.7773684,
            0.8098109,
            0.84042233,
            0.86897135,
            0.8952358,
            0.91901,
            0.94010323,
            0.9583462,
            0.97358865,
            0.9857061,
            0.9954344,
            1.0,
        ]
    )

    return bk


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
    if ptop < ptop_min:
        ak1 = (constants.KAPPA + 1.0) / constants.KAPPA
        peln[:, :, 0] = peln[:, :, 1] - ak1
    else:
        peln[:, :, 0] = np.log(ptop)


def geopotential_perturbation(lat, eta_value):
    """
    Equation (7) JRMS2006, just the perturbation component
    """
    u_comp = u0 * (np.cos(eta_value) ** (3.0 / 2.0))
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
    t_mean = t_0 * eta[:] ** (constants.RDGAS * lapse_rate / constants.GRAV)
    # above troposphere
    t_mean[eta_tropopause > eta] = (
        t_mean[eta_tropopause > eta]
        + delta_t * (eta_tropopause - eta[eta_tropopause > eta]) ** 5.0
    )
    return t_mean


def _initialize_delp(ak, bk, ps, shape):
    delp = np.zeros(shape)
    delp[:, :, :-1] = (
        ak[None, None, 1:]
        - ak[None, None, :-1]
        + ps[:, :, None] * (bk[None, None, 1:] - bk[None, None, :-1])
    )

    return delp


def initialize_delp(ps, ak, bk):
    return (
        ak[None, None, 1:]
        - ak[None, None, :-1]
        + ps[:, :, None] * (bk[None, None, 1:] - bk[None, None, :-1])
    )


def initialize_delz(pt, peln):
    return constants.RDG * pt[:, :, :-1] * (peln[:, :, 1:] - peln[:, :, :-1])


def _initialize_delz_w(pe, ps, pt, qvapor, tc_properties, calc, shape):

    delz = np.zeros(shape)
    w = np.zeros(shape)
    delz[:, :, :-1] = (
        constants.RDGAS
        * pt[:, :, :-1]
        * (1 + constants.ZVIR * qvapor[:, :, :-1])
        / constants.GRAV
        * np.log(pe[:, :, :-1] / pe[:, :, 1:])
    )

    return delz, w


def _initialize_edge_pressure(delp, ptop, shape):
    pe = np.zeros(shape)
    pe[:, :, 0] = ptop
    for k in range(1, pe.shape[2]):
        pe[:, :, k] = ptop + np.sum(delp[:, :, :k], axis=2)
    return pe


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


def _initialize_qvapor_temperature(grid_data, pe, ps, tc_properties, calc, shape):

    qvapor = np.zeros(shape)
    pt = np.zeros(shape)
    height = np.zeros(shape)

    ptmp = 0.5 * (pe[:, :, :-1] + pe[:, :, 1:])
    height[:, :, :-1] = (calc["t00"] / tc_properties["gamma"]) * (
        1.0 - (ptmp / ps[:, :, None]) ** calc["exponent"]
    )
    qvapor = (
        tc_properties["q00"]
        * np.exp(-height / tc_properties["zq1"])
        * np.exp(-((height / tc_properties["zq2"]) ** tc_properties["exppz"]))
    )

    p2 = np.transpose(
        np.stack(
            [
                grid_data._horizontal_data.lon_agrid.data,
                grid_data._horizontal_data.lat_agrid.data,
            ]
        ),
        [1, 2, 0],
    )
    r = great_circle_distance_lon_lat(
        calc["p0"][0], p2[:, :, 0], calc["p0"][1], p2[:, :, 1], constants.RADIUS, np
    )

    pt = _calculate_pt_height(height, qvapor, r, tc_properties, calc)

    qvapor[height > tc_properties["ztrop"]] = tc_properties["qtrop"]
    pt[height > tc_properties["ztrop"]] = calc["ttrop"]

    return qvapor, pt


def _initialize_vortex_ps_phis(grid_data, shape, tc_properties, calc):
    p0 = [np.deg2rad(tc_properties["lon_tc"]), np.deg2rad(tc_properties["lat_tc"])]

    phis = np.zeros(shape[:2])
    ps = np.zeros(shape[:2])
    # breakpoint()
    grid = np.transpose(
        np.stack(
            [
                grid_data._horizontal_data.lon_agrid.data,
                grid_data._horizontal_data.lat_agrid.data,
            ]
        ),
        [1, 2, 0],
    )
    ps = _calculate_vortex_surface_pressure_with_radius(calc["p0"], grid, tc_properties)

    grid = np.transpose(
        np.stack(
            [grid_data._horizontal_data.lon.data, grid_data._horizontal_data.lat.data]
        ),
        [1, 2, 0],
    )
    ps_vc = np.zeros(shape[:2])
    p_grid = 0.5 * (grid[:, :-1, :] + grid[:, 1:, :])
    ps_vc[:, :-1] = _calculate_vortex_surface_pressure_with_radius(
        p0, p_grid, tc_properties
    )

    ps_uc = np.zeros(shape[:2])
    p_grid = 0.5 * (grid[:-1, :, :] + grid[1:, :, :])
    ps_uc[:-1, :] = _calculate_vortex_surface_pressure_with_radius(
        p0, p_grid, tc_properties
    )

    output_dict = {"ps": ps, "ps_uc": ps_uc, "ps_vc": ps_vc, "phis": phis}

    return output_dict


def _initialize_wind_dgrid(
    grid_data, tc_properties, calc, pe_u, pe_v, ps_u, ps_v, shape
):
    # u-wind
    ud = np.zeros(shape)

    grid = np.transpose(
        np.stack(
            [grid_data._horizontal_data.lon.data, grid_data._horizontal_data.lat.data]
        ),
        [1, 2, 0],
    )
    p1 = grid[:-1, :, :]
    p2 = grid[1:, :, :]
    muv = _find_midpoint_unit_vectors(p1, p2)
    dist = _calculate_distance_from_tc_center(pe_u, ps_u, muv, calc, tc_properties)

    utmp = _calculate_utmp(dist["height"][:-1, :, :], dist, calc, tc_properties)
    vtmp = utmp * dist["d2"][:, :, None]
    print()
    utmp = utmp * dist["d1"][:, :, None]

    ud[:-1, :, :-1] = (
        utmp * np.sum(muv["unit_dir"] * muv["exv"], 2)[:, :, None]
        + vtmp * np.sum(muv["unit_dir"] * muv["eyv"], 2)[:, :, None]
    )
    ud[:, :, :-1][dist["height"] > tc_properties["ztrop"]] = 0

    # v-wind
    vd = np.zeros(shape)
    p1 = grid[:, :-1, :]
    p2 = grid[:, 1:, :]
    muv = _find_midpoint_unit_vectors(p1, p2)
    dist = _calculate_distance_from_tc_center(pe_v, ps_v, muv, calc, tc_properties)

    utmp = _calculate_utmp(dist["height"][:, :-1, :], dist, calc, tc_properties)
    vtmp = utmp * dist["d2"][:, :, None]
    utmp *= dist["d1"][:, :, None]

    vd[:, :-1, :-1] = (
        utmp * np.sum(muv["unit_dir"] * muv["exv"], 2)[:, :, None]
        + vtmp * np.sum(muv["unit_dir"] * muv["eyv"], 2)[:, :, None]
    )
    vd[:, :, :-1][dist["height"] > tc_properties["ztrop"]] = 0

    return ud, vd


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


def _interpolate_winds_dgrid_agrid(grid_data, ud, vd, tc_properties, shape):

    ua = np.zeros(shape)
    va = np.zeros(shape)
    if tc_properties["vort"] is True:
        ua[:, :-1, :] = (
            0.5
            * (
                ud[:, :-1, :] * grid_data._horizontal_data.dx.data[:, :-1, None]
                + ud[:, 1:, :] * grid_data._horizontal_data.dx.data[:, 1:, None]
            )
            / grid_data.dxa.data[:, :-1, None]
        )
        va[:-1, :, :] = (
            0.5
            * (
                vd[:-1, :, :] * grid_data._horizontal_data.dy.data[:-1, :, None]
                + vd[1:, :, :] * grid_data._horizontal_data.dy.data[1:, :, None]
            )
            / grid_data._horizontal_data.dya.data[:-1, :, None]
        )
    else:
        pass

    # TODO translate the not vort case#
    #             do i=isd,ied
    #             tmp1j(:) = 0.0
    #             tmp2j(:) = uin(i,:)*dyc(i,:)
    #             tmp3j(:) = dyc(i,:)
    #             call interp_left_edge_1d(tmp1j, tmp2j, tmp3j, jsd, jed+1, interpOrder)
    #             uout(i,jsd:jed) = tmp1j(jsd+1:jed+1)/dya(i,jsd:jed)
    #          enddo
    #          do j=jsd,jed
    #             tmp1i(:) = 0.0
    #             tmp2i(:) = vin(:,j)*dxc(:,j)
    #             tmp3i(:) = dxc(:,j)
    #             call interp_left_edge_1d(tmp1i, tmp2i, tmp3i, isd, ied+1, interpOrder)
    #             vout(isd:ied,j) = tmp1i(isd+1:ied+1)/dxa(isd:ied,j)
    #          enddo #2934

    return ua, va


def local_compute_size(data_array_shape):
    nx = data_array_shape[0] - 2 * nhalo - 1
    ny = data_array_shape[1] - 2 * nhalo - 1
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
    ps[:] = surface_pressure
    delp[:, :, :-1] = initialize_delp(ps, ak, bk)
    pe[:] = initialize_edge_pressure(delp, ptop)
    peln[:] = initialize_log_pressure_interfaces(pe, ptop)
    pk[:], pkz[:] = initialize_kappa_pressures(pe, peln, ptop)
    eta[:-1], eta_v[:-1] = compute_eta(ak, bk)


def _some_inital_calculations(tc_properties):
    t00 = tc_properties["Ts0"] * (1.0 + constants.ZVIR * tc_properties["q00"])  # num
    p0 = [np.deg2rad(tc_properties["lon_tc"]), np.deg2rad(tc_properties["lat_tc"])]
    exponent = constants.RDGAS * tc_properties["gamma"] / constants.GRAV  # num
    cor = 2.0 * constants.OMEGA * np.sin(np.deg2rad(tc_properties["lat_tc"]))  # num
    ttrop = t00 - tc_properties["gamma"] * tc_properties["ztrop"]

    calc = {
        "cor": cor,
        "exponent": exponent,
        "p0": p0,
        "ttrop": ttrop,
        "t00": t00,
    }

    return calc


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
    ptmp = delp[:, :, :-1] / (peln[:, :, 1:] - peln[:, :, :-1]) - surface_pressure
    # Similar to equation 18 of DCMIP2016 without a cutoff at tropopause
    return (
        q0
        * np.exp(-((lat_agrid[:, :, None] / pcen[1]) ** 4.0))
        * np.exp(-((ptmp / pw) ** 2.0))
    )


def surface_geopotential_perturbation(lat):
    """
    From JRMS2006:
    * 'In hydrostatic models with pressure-based vertical coordinates, it's
       only necessary to initialize surface geopotential.'
    * 'balances the non-zero zonal wind at the surface with surface elevation zs'
    """
    surface_level = vertical_coordinate(eta_surface)
    return geopotential_perturbation(lat, surface_level)


def temperature(eta, eta_v, t_mean, lat):
    """
    Equation (6)JRMS2006
    The total temperature distribution from the horizontal-mean temperature
     and a horizontal variation at each level
    """
    lat = lat[:, :, None]
    return t_mean + 0.75 * (eta[:] * math.pi * u0 / constants.RDGAS) * np.sin(
        eta_v[:]
    ) * np.sqrt(np.cos(eta_v[:])) * (
        (-2.0 * (np.sin(lat) ** 6.0) * (np.cos(lat) ** 2.0 + 1.0 / 3.0) + 10.0 / 63.0)
        * 2.0
        * u0
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
    return (eta_value - eta_0) * math.pi * 0.5


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
    u_component[slice_3d] = local_coordinate_transformation(
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
    return u0 * np.cos(eta_v[:]) ** (3.0 / 2.0) * np.sin(2.0 * lat[:, :, None]) ** 2.0
