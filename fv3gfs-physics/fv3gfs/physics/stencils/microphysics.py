import copy

import numpy as np
from gt4py import storage
from gt4py.gtscript import (
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    horizontal,
    interval,
)

import fv3gfs.util
import fv3gfs.util.pace.gt4py_utils as utils
from fv3gfs.physics.functions.microphysics_funcs import *
from fv3gfs.physics.global_constants import *
from fv3gfs.util.pace.stencil import StencilFactory
from fv3gfs.util.pace.typing import Float, FloatField, FloatFieldIJ, Int, IntField
from fv3gfs.util.quantity import Quantity


def fields_init(
    land: FloatField,
    area: FloatFieldIJ,
    h_var: FloatField,
    rh_adj: FloatField,
    rh_rain: FloatField,
    graupel: FloatField,
    ice: FloatField,
    rain: FloatField,
    snow: FloatField,
    qa: FloatField,
    qg: FloatField,
    qi: FloatField,
    ql: FloatField,
    qn: FloatField,
    qr: FloatField,
    qs: FloatField,
    qv: FloatField,
    pt: FloatField,
    delp: FloatField,
    dz: FloatField,
    qgz: FloatField,
    qiz: FloatField,
    qlz: FloatField,
    qrz: FloatField,
    qsz: FloatField,
    qvz: FloatField,
    tz: FloatField,
    qi_dt: FloatField,
    qs_dt: FloatField,
    uin: FloatField,
    vin: FloatField,
    qa0: FloatField,
    qg0: FloatField,
    qi0: FloatField,
    ql0: FloatField,
    qr0: FloatField,
    qs0: FloatField,
    qv0: FloatField,
    t0: FloatField,
    dp0: FloatField,
    den0: FloatField,
    dz0: FloatField,
    u0: FloatField,
    v0: FloatField,
    dp1: FloatField,
    p1: FloatField,
    u1: FloatField,
    v1: FloatField,
    ccn: FloatField,
    c_praut: FloatField,
    use_ccn: Int,
    c_air: Float,
    c_vap: Float,
    d0_vap: Float,
    lv00: Float,
    dt_in: Float,
    rdt: Float,
    cpaut: Float,
):

    with computation(PARALLEL), interval(...):

        # Initialize precipitation
        graupel = 0.0
        rain = 0.0
        snow = 0.0
        ice = 0.0

        # This is to prevent excessive build-up of cloud ice from
        # external sources
        if de_ice == 1:

            qio = qi - dt_in * qi_dt  # Orginal qi before phys
            qin = max(qio, qi0_max)  # Adjusted value

            if qi > qin:

                qs = qs + qi - qin
                qi = qin

                dqi = (qin - qio) * rdt  # Modified qi tendency
                qs_dt = qs_dt + qi_dt - dqi
                qi_dt = dqi

        qiz = qi
        qsz = qs

        t0 = pt
        tz = t0
        dp1 = delp
        dp0 = dp1  # Moist air mass * grav

        # Convert moist mixing ratios to dry mixing ratios
        qvz = qv
        qlz = ql
        qrz = qr
        qgz = qg

        dp1 = dp1 * (1.0 - qvz)
        omq = dp0 / dp1

        qvz = qvz * omq
        qlz = qlz * omq
        qrz = qrz * omq
        qiz = qiz * omq
        qsz = qsz * omq
        qgz = qgz * omq

        qa0 = qa
        dz0 = dz

        den0 = -dp1 / (grav * dz0)  # Density of dry air
        p1 = den0 * rdgas * t0  # Dry air pressure

        # Save a copy of old values for computing tendencies
        qv0 = qvz
        ql0 = qlz
        qr0 = qrz
        qi0 = qiz
        qs0 = qsz
        qg0 = qgz

        # For sedi_momentum
        u0 = uin
        v0 = vin
        u1 = u0
        v1 = v0

        if prog_ccn == 1:

            # Convert #/cc to #/m^3
            ccn = qn * 1.0e6
            c_praut = cpaut * (ccn * rhor) ** (-1.0 / 3.0)

        else:

            ccn = (ccn_l * land + ccn_o * (1.0 - land)) * 1.0e6

    with computation(BACKWARD):

        with interval(-1, None):

            if (prog_ccn == 0) and (use_ccn == 1):

                # ccn is formulted as ccn = ccn_surface * (den / den_surface)
                ccn = ccn * rdgas * tz / p1

        with interval(0, -1):

            if (prog_ccn == 0) and (use_ccn == 1):

                # Propagate downwards previously computed values of ccn
                ccn = ccn[0, 0, +1]

    with computation(PARALLEL), interval(...):

        if prog_ccn == 0:
            c_praut = cpaut * (ccn * rhor) ** (-1.0 / 3.0)

        # Calculate horizontal subgrid variability
        # Total water subgrid deviation in horizontal direction
        # Default area dependent form: use dx ~ 100 km as the base
        s_leng = sqrt(sqrt(area * 1.0e-10))
        t_land = dw_land * s_leng
        t_ocean = dw_ocean * s_leng
        h_var = t_land * land + t_ocean * (1.0 - land)
        h_var = min(0.2, max(0.01, h_var))

        # Relative humidity increment
        rh_adj = 1.0 - h_var - rh_inc
        rh_rain = max(0.35, rh_adj - rh_inr)

        # Fix all negative water species
        if fix_negative == 1:

            # Define heat capacity and latent heat coefficient
            cvm = c_air + qvz * c_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice
            lcpk = (lv00 + d0_vap * tz) / cvm
            icpk = (li00 + dc_ice * tz) / cvm

            # Ice phase

            # If cloud ice < 0, borrow from snow
            if qiz < 0.0:

                qsz = qsz + qiz
                qiz = 0.0

            # If snow < 0, borrow from graupel
            if qsz < 0.0:

                qgz = qgz + qsz
                qsz = 0.0

            # If graupel < 0, borrow from rain
            if qgz < 0.0:

                qrz = qrz + qgz
                tz = tz - qgz * icpk  # Heating
                qgz = 0.0

            # Liquid phase

            # If rain < 0, borrow from cloud water
            if qrz < 0.0:

                qlz = qlz + qrz
                qrz = 0.0

            # If cloud water < 0, borrow from water vapor
            if qlz < 0.0:

                qvz = qvz + qlz
                tz = tz - qlz * lcpk  # Heating
                qlz = 0.0

    with computation(FORWARD), interval(1, None):

        # Fix water vapor; borrow from below
        if (fix_negative == 1) and (qvz[0, 0, -1] < 0.0):
            qvz[0, 0, 0] = qvz[0, 0, 0] + qvz[0, 0, -1] * dp1[0, 0, -1] / dp1[0, 0, 0]

    with computation(PARALLEL), interval(0, -1):

        if (fix_negative == 1) and (qvz < 0.0):
            qvz = 0.0

    # Bottom layer; borrow from above
    with computation(PARALLEL):

        with interval(-2, -1):

            flag = 0

            if (fix_negative == 1) and (qvz[0, 0, +1] < 0.0) and (qvz > 0.0):

                dq = min(-qvz[0, 0, +1] * dp1[0, 0, +1], qvz[0, 0, 0] * dp1[0, 0, 0])
                flag = 1

        with interval(-1, None):

            flag = 0

            if (fix_negative == 1) and (qvz < 0.0) and (qvz[0, 0, -1] > 0.0):

                dq = min(-qvz[0, 0, 0] * dp1[0, 0, 0], qvz[0, 0, -1] * dp1[0, 0, -1])
                flag = 1

    with computation(PARALLEL):

        with interval(-2, -1):

            if flag == 1:

                qvz = qvz - dq / dp1

        with interval(-1, None):

            if flag == 1:

                qvz = qvz + dq / dp1


def warm_rain(
    h_var: FloatField,
    rain: FloatField,
    qgz: FloatField,
    qiz: FloatField,
    qlz: FloatField,
    qrz: FloatField,
    qsz: FloatField,
    qvz: FloatField,
    tz: FloatField,
    den: FloatField,
    denfac: FloatField,
    w: FloatField,
    t0: FloatField,
    den0: FloatField,
    dz0: FloatField,
    dz1: FloatField,
    dp1: FloatField,
    m1: FloatField,
    vtrz: FloatField,
    ccn: FloatField,
    c_praut: FloatField,
    m1_sol: FloatField,
    m2_rain: FloatField,
    m2_sol: FloatField,
    is_first: Int,
    do_sedi_w: Int,
    p_nonhydro: Int,
    use_ccn: Int,
    c_air: Float,
    c_vap: Float,
    d0_vap: Float,
    lv00: Float,
    fac_rc: Float,
    cracw: Float,
    crevp_0: Float,
    crevp_1: Float,
    crevp_2: Float,
    crevp_3: Float,
    crevp_4: Float,
    t_wfr: Float,
    so3: Float,
    dt_rain: Float,
    zs: Float,
):

    with computation(PARALLEL), interval(...):

        if is_first == 1:

            # Define air density based on hydrostatical property
            if p_nonhydro == 1:

                dz1 = dz0
                den = den0  # Dry air density remains the same
                denfac = sqrt(sfcrho / den)

            else:

                dz1 = dz0 * tz / t0  # Hydrostatic balance
                den = den0 * dz0 / dz1
                denfac = sqrt(sfcrho / den)

        # Time-split warm rain processes: 1st pass
        dt5 = 0.5 * dt_rain

        # Terminal speed of rain
        m1_rain = 0.0

    with computation(FORWARD):

        with interval(0, 1):

            if qrz > qrmin:
                no_fall = 0
            else:
                no_fall = 1

        with interval(1, None):

            if no_fall[0, 0, -1] == 1:

                if qrz > qrmin:
                    no_fall = 0
                else:
                    no_fall = 1

            else:

                no_fall = 0

    with computation(BACKWARD), interval(0, -1):

        if no_fall[0, 0, +1] == 0:
            no_fall = no_fall[0, 0, +1]

    with computation(PARALLEL), interval(...):

        vtrz, r1 = compute_rain_fspeed(no_fall, qrz, den)

    with computation(BACKWARD):

        with interval(-1, None):

            if no_fall == 0:
                ze = zs - dz1

        with interval(0, -1):

            if no_fall == 0:
                ze = ze[0, 0, +1] - dz1  # dz < 0

    with computation(PARALLEL), interval(...):

        if no_fall == 0:

            # Evaporation and accretion of rain for the first 1/2 time step
            qgz, qiz, qlz, qrz, qsz, qvz, tz = revap_racc(
                dt5,
                c_air,
                c_vap,
                d0_vap,
                lv00,
                t_wfr,
                cracw,
                crevp_0,
                crevp_1,
                crevp_2,
                crevp_3,
                crevp_4,
                h_var,
                qgz,
                qiz,
                qlz,
                qrz,
                qsz,
                qvz,
                tz,
                den,
                denfac,
            )

            if do_sedi_w == 1:
                dm = dp1 * (1.0 + qvz + qlz + qrz + qiz + qsz + qgz)

    # Mass flux induced by falling rain
    with computation(PARALLEL):

        with interval(0, 1):

            if (use_ppm == 1) and (no_fall == 0):
                zt = ze

        with interval(1, -1):

            if (use_ppm == 1) and (no_fall == 0):
                zt = ze - dt5 * (vtrz[0, 0, -1] + vtrz)

        with interval(-1, None):

            if (use_ppm == 1) and (no_fall == 0):

                zt = ze - dt5 * (vtrz[0, 0, -1] + vtrz)

                zt_kbot1 = zs - dt_rain * vtrz

    with computation(FORWARD):

        with interval(1, -1):

            if (use_ppm == 1) and (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                zt = zt[0, 0, -1] - dz_min

        with interval(-1, None):

            if use_ppm == 1:

                if (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                    zt = zt[0, 0, -1] - dz_min

                if (no_fall == 0) and (zt_kbot1 >= zt):
                    zt_kbot1 = zt - dz_min

    with computation(BACKWARD), interval(0, -1):

        if (use_ppm == 1) and (no_fall == 0):
            zt_kbot1 = zt_kbot1[0, 0, +1]

    with computation(PARALLEL):

        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - ze[0, 0, +1]

        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - zs

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            dd = dt_rain * vtrz
            qrz = qrz * dp1

    # Sedimentation
    with computation(FORWARD):

        with interval(0, 1):

            if (use_ppm == 0) and (no_fall == 0):
                qm = qrz / (dz + dd)

        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                qm = (qrz[0, 0, 0] + dd[0, 0, -1] * qm[0, 0, -1]) / (dz + dd)

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            # qm is density at this stage
            qm = qm * dz

    # Output mass fluxes
    with computation(FORWARD):

        with interval(0, 1):

            if (use_ppm == 0) and (no_fall == 0):
                m1_rain = qrz - qm

        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                m1_rain = m1_rain[0, 0, -1] + qrz[0, 0, 0] - qm

    with computation(BACKWARD):

        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                r1 = m1_rain

        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                r1 = r1[0, 0, +1]

    with computation(PARALLEL):

        with interval(0, 1):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qrz = qm / dp1

                # Vertical velocity transportation during sedimentation
                if do_sedi_w == 1:
                    w = (dm * w + m1_rain * vtrz) / (dm - m1_rain)

        with interval(1, None):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qrz = qm / dp1

                # Vertical velocity transportation during sedimentation
                if do_sedi_w == 1:

                    w[0, 0, 0] = (
                        dm * w[0, 0, 0]
                        - m1_rain[0, 0, -1] * vtrz[0, 0, -1]
                        + m1_rain * vtrz
                    ) / (dm + m1_rain[0, 0, -1] - m1_rain)

    # Heat transportation during sedimentation
    with computation(PARALLEL):

        with interval(0, 1):

            if (do_sedi_heat == 1) and (no_fall == 0):

                # Input q fields are dry mixing ratios, and dm is dry air mass
                dgz = -0.5 * grav * dz1
                cvn = dp1 * (
                    cv_air
                    + qvz * cv_vap
                    + (qrz + qlz) * c_liq
                    + (qiz + qsz + qgz) * c_ice
                )

                # - Assumption: The ke in the falling condensates is negligible
                #               compared to the potential energy that was
                #               unaccounted for. Local thermal equilibrium is
                #               assumed, and the loss in pe is transformed into
                #               internal energy (to heat the whole grid box).
                # - Backward time-implicit upwind transport scheme:
                # - dm here is dry air mass
                tmp = cvn + m1_rain * c_liq
                tz = tz + m1_rain * dgz / tmp

        with interval(1, None):

            if (do_sedi_heat == 1) and (no_fall == 0):

                # Input q fields are dry mixing ratios, and dm is dry air mass
                dgz = -0.5 * grav * dz1
                cvn = dp1 * (
                    cv_air
                    + qvz * cv_vap
                    + (qrz + qlz) * c_liq
                    + (qiz + qsz + qgz) * c_ice
                )

    # Implicit algorithm
    with computation(FORWARD), interval(1, None):

        if (do_sedi_heat == 1) and (no_fall == 0):

            tz[0, 0, 0] = (
                (cvn + c_liq * (m1_rain - m1_rain[0, 0, -1])) * tz[0, 0, 0]
                + m1_rain[0, 0, -1] * c_liq * tz[0, 0, -1]
                + dgz * (m1_rain[0, 0, -1] + m1_rain)
            ) / (cvn + c_liq * m1_rain)

    with computation(PARALLEL), interval(...):

        if no_fall == 0:

            # Evaporation and accretion of rain for the remaining 1/2 time step
            qgz, qiz, qlz, qrz, qsz, qvz, tz = revap_racc(
                dt5,
                c_air,
                c_vap,
                d0_vap,
                lv00,
                t_wfr,
                cracw,
                crevp_0,
                crevp_1,
                crevp_2,
                crevp_3,
                crevp_4,
                h_var,
                qgz,
                qiz,
                qlz,
                qrz,
                qsz,
                qvz,
                tz,
                den,
                denfac,
            )

        # Auto-conversion assuming linear subgrid vertical distribution of
        # cloud water following lin et al. 1994, mwr
        if irain_f != 0:

            qlz, qrz = autoconv_no_subgrid_var(
                use_ccn, fac_rc, t_wfr, so3, dt_rain, qlz, qrz, tz, den, ccn, c_praut
            )

    # With subgrid variability
    with computation(FORWARD):

        with interval(0, 1):

            if (irain_f == 0) and (z_slope_liq == 1):
                dl = 0.0

        with interval(1, None):

            if (irain_f == 0) and (z_slope_liq == 1):
                dq = 0.5 * (qlz[0, 0, 0] - qlz[0, 0, -1])

    with computation(PARALLEL):

        with interval(1, -1):

            if (irain_f == 0) and (z_slope_liq == 1):

                # Use twice the strength of the positive definiteness limiter (lin et al 1994)
                dl = 0.5 * min(abs(dq + dq[0, 0, +1]), 0.5 * qlz[0, 0, 0])

                if dq * dq[0, 0, +1] <= 0.0:

                    if dq > 0.0:  # Local maximum

                        dl = min(dl, min(dq, -dq[0, 0, +1]))

                    else:

                        dl = 0.0

        with interval(-1, None):

            if (irain_f == 0) and (z_slope_liq == 1):
                dl = 0.0

    with computation(PARALLEL), interval(...):

        if irain_f == 0:

            if z_slope_liq == 1:

                # Impose a presumed background horizontal variability that is
                # proportional to the value itself
                dl = max(dl, max(qvmin, h_var * qlz))

            else:

                dl = max(qvmin, h_var * qlz)

            qlz, qrz = autoconv_subgrid_var(
                use_ccn,
                fac_rc,
                t_wfr,
                so3,
                dt_rain,
                qlz,
                qrz,
                tz,
                den,
                ccn,
                c_praut,
                dl,
            )

        rain = rain + r1
        m2_rain = m2_rain + m1_rain

        if is_first == 1:

            m1 = m1 + m1_rain

        else:

            m2_sol = m2_sol + m1_sol
            m1 = m1 + m1_rain + m1_sol


def sedimentation(
    graupel: FloatField,
    ice: FloatField,
    rain: FloatField,
    snow: FloatField,
    qgz: FloatField,
    qiz: FloatField,
    qlz: FloatField,
    qrz: FloatField,
    qsz: FloatField,
    qvz: FloatField,
    tz: FloatField,
    den: FloatField,
    w: FloatField,
    dz1: FloatField,
    dp1: FloatField,
    vtgz: FloatField,
    vtsz: FloatField,
    m1_sol: FloatField,
    do_sedi_w: Int,
    c_air: Float,
    c_vap: Float,
    d0_vap: Float,
    lv00: Float,
    log_10: Float,
    zs: Float,
    dts: Float,
    fac_imlt: Float,
):

    with computation(PARALLEL), interval(...):

        # Sedimentation of cloud ice, snow, and graupel
        vtgz, vtiz, vtsz = fall_speed(log_10, qgz, qiz, qlz, qsz, tz, den)

        dt5 = 0.5 * dts

        # Define heat capacity and latent heat coefficient
        m1_sol = 0.0

        lhi = li00 + dc_ice * tz
        q_liq = qlz + qrz
        q_sol = qiz + qsz + qgz
        cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
        icpk = lhi / cvm

    # Find significant melting level
    """
    k0 removed to avoid having to introduce a k_idx field
    """
    with computation(FORWARD):

        with interval(0, 1):

            if tz > tice:
                stop_k = 1
            else:
                stop_k = 0

        with interval(1, -1):

            if stop_k[0, 0, -1] == 0:

                if tz > tice:
                    stop_k = 1
                else:
                    stop_k = 0

            else:

                stop_k = 1

        with interval(-1, None):

            stop_k = 1

    with computation(PARALLEL), interval(...):

        if stop_k == 1:

            # Melting of cloud ice (before fall)
            tc = tz - tice

            if (qiz > qcmin) and (tc > 0.0):

                sink = min(qiz, fac_imlt * tc / icpk)
                tmp = min(sink, dim(ql_mlt, qlz))
                qlz = qlz + tmp
                qrz = qrz + sink - tmp
                qiz = qiz - sink
                q_liq = q_liq + sink
                q_sol = q_sol - sink
                cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
                tz = tz - sink * lhi / cvm
                tc = tz - tice

    with computation(PARALLEL), interval(0, -1):

        # Turn off melting when cloud microphysics time step is small
        if dts < 60.0:
            stop_k = 0

        # sjl, turn off melting of falling cloud ice, snow and graupel
        stop_k = 0

    with computation(BACKWARD):

        with interval(-1, None):

            ze = zs - dz1

        with interval(1, -1):

            ze = ze[0, 0, +1] - dz1  # dz < 0

        with interval(0, 1):

            ze = ze[0, 0, +1] - dz1  # dz < 0
            zt = ze

    with computation(PARALLEL), interval(...):

        if stop_k == 1:

            # Update capacity heat and latent heat coefficient
            lhi = li00 + dc_ice * tz
            icpk = lhi / cvm

    # Melting of falling cloud ice into rain
    with computation(FORWARD):

        with interval(0, 1):

            if qiz > qrmin:
                no_fall = 0
            else:
                no_fall = 1

        with interval(1, None):

            if no_fall[0, 0, -1] == 1:

                if qiz > qrmin:
                    no_fall = 0
                else:
                    no_fall = 1

            else:

                no_fall = 0

    with computation(BACKWARD), interval(0, -1):

        if no_fall[0, 0, +1] == 0:
            no_fall = no_fall[0, 0, +1]

    with computation(PARALLEL), interval(...):

        if (vi_fac < 1.0e-5) or (no_fall == 1):
            i1 = 0.0

    with computation(PARALLEL):

        with interval(1, -1):

            if (vi_fac >= 1.0e-5) and (no_fall == 0):
                zt = ze - dt5 * (vtiz[0, 0, -1] + vtiz)

        with interval(-1, None):

            if (vi_fac >= 1.0e-5) and (no_fall == 0):

                zt = ze - dt5 * (vtiz[0, 0, -1] + vtiz)
                zt_kbot1 = zs - dts * vtiz

    with computation(FORWARD):

        with interval(1, -1):

            if (vi_fac >= 1.0e-5) and (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                zt = zt[0, 0, -1] - dz_min

        with interval(-1, None):

            if (vi_fac >= 1.0e-5) and (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                zt = zt[0, 0, -1] - dz_min

            if (vi_fac >= 1.0e-5) and (no_fall == 0) and (zt_kbot1 >= zt):
                zt_kbot1 = zt - dz_min

    with computation(BACKWARD), interval(0, -1):

        if (vi_fac >= 1.0e-5) and (no_fall == 0):
            zt_kbot1 = zt_kbot1[0, 0, +1] - dz_min

    with computation(PARALLEL), interval(...):

        if (vi_fac >= 1.0e-5) and (no_fall == 0):

            if do_sedi_w == 1:
                dm = dp1 * (1.0 + qvz + qlz + qrz + qiz + qsz + qgz)

    with computation(PARALLEL):

        with interval(0, -1):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                dz = ze - ze[0, 0, +1]

        with interval(-1, None):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                dz = ze - zs

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):

            dd = dts * vtiz
            qiz = qiz * dp1

    # Sedimentation
    with computation(FORWARD):

        with interval(0, 1):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                qm = qiz / (dz + dd)

        with interval(1, None):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                qm = (qiz[0, 0, 0] + dd[0, 0, -1] * qm[0, 0, -1]) / (dz + dd)

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):

            # qm is density at this stage
            qm = qm * dz

    # Output mass fluxes
    with computation(FORWARD):

        with interval(0, 1):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                m1_sol = qiz - qm

        with interval(1, None):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                m1_sol = m1_sol[0, 0, -1] + qiz[0, 0, 0] - qm

    with computation(BACKWARD):

        with interval(-1, None):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                i1 = m1_sol

        with interval(0, -1):

            if (use_ppm == 0) and (vi_fac >= 1.0e-5) and (no_fall == 0):
                i1 = i1[0, 0, +1]

    with computation(PARALLEL):

        with interval(0, 1):

            if (vi_fac >= 1.0e-5) and (no_fall == 0):

                if use_ppm == 0:

                    # Update
                    qiz = qm / dp1

                # Vertical velocity transportation during sedimentation
                if do_sedi_w == 1:
                    w = (dm * w + m1_sol * vtiz) / (dm - m1_sol)

        with interval(1, None):

            if (vi_fac >= 1.0e-5) and (no_fall == 0):

                if use_ppm == 0:

                    # Update
                    qiz = qm / dp1

                if do_sedi_w == 1:

                    w[0, 0, 0] = (
                        dm * w[0, 0, 0]
                        - m1_sol[0, 0, -1] * vtiz[0, 0, -1]
                        + m1_sol * vtiz
                    ) / (dm + m1_sol[0, 0, -1] - m1_sol)

    # Melting of falling snow into rain
    with computation(FORWARD):

        with interval(0, 1):

            if qsz > qrmin:
                no_fall = 0
            else:
                no_fall = 1

        with interval(1, None):

            if no_fall[0, 0, -1] == 1:

                if qsz > qrmin:
                    no_fall = 0
                else:
                    no_fall = 1

            else:

                no_fall = 0

    with computation(BACKWARD), interval(0, -1):

        if no_fall[0, 0, +1] == 0:
            no_fall = no_fall[0, 0, +1]

    with computation(PARALLEL), interval(...):

        r1 = 0.0

        if no_fall == 1:
            s1 = 0.0

    with computation(PARALLEL):

        with interval(1, -1):

            if no_fall == 0:
                zt = ze - dt5 * (vtsz[0, 0, -1] + vtsz)

        with interval(-1, None):

            if no_fall == 0:

                zt = ze - dt5 * (vtsz[0, 0, -1] + vtsz)
                zt_kbot1 = zs - dts * vtsz

    with computation(FORWARD):

        with interval(1, -1):

            if (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                zt = zt[0, 0, -1] - dz_min

        with interval(-1, None):

            if (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                zt = zt[0, 0, -1] - dz_min

            if (no_fall == 0) and (zt_kbot1 >= zt):
                zt_kbot1 = zt - dz_min

    with computation(BACKWARD), interval(0, -1):

        if no_fall == 0:
            zt_kbot1 = zt_kbot1[0, 0, +1] - dz_min

    with computation(PARALLEL), interval(...):

        if no_fall == 0:

            if do_sedi_w == 1:
                dm = dp1 * (1.0 + qvz + qlz + qrz + qiz + qsz + qgz)

    with computation(PARALLEL):

        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - ze[0, 0, +1]

        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - zs

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            dd = dts * vtsz
            qsz = qsz * dp1

    # Sedimentation
    with computation(FORWARD):

        with interval(0, 1):

            if (use_ppm == 0) and (no_fall == 0):
                qm = qsz / (dz + dd)

        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                qm = (qsz[0, 0, 0] + dd[0, 0, -1] * qm[0, 0, -1]) / (dz + dd)

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            # qm is density at this stage
            qm = qm * dz

    # Output mass fluxes
    with computation(FORWARD):

        with interval(0, 1):

            if (use_ppm == 0) and (no_fall == 0):
                m1_tf = qsz - qm

        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                m1_tf = m1_tf[0, 0, -1] + qsz[0, 0, 0] - qm

    with computation(BACKWARD):

        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                s1 = m1_tf

        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                s1 = s1[0, 0, +1]

    with computation(PARALLEL):

        with interval(0, 1):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qsz = qm / dp1

                m1_sol = m1_sol + m1_tf

                # Vertical velocity transportation during sedimentation
                if do_sedi_w == 1:
                    w = (dm * w + m1_tf * vtsz) / (dm - m1_tf)

        with interval(1, None):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qsz = qm / dp1

                m1_sol = m1_sol + m1_tf

                if do_sedi_w == 1:

                    w[0, 0, 0] = (
                        dm * w[0, 0, 0]
                        - m1_tf[0, 0, -1] * vtsz[0, 0, -1]
                        + m1_tf * vtsz
                    ) / (dm + m1_tf[0, 0, -1] - m1_tf)

    # Melting of falling graupel into rain
    with computation(FORWARD):

        with interval(0, 1):

            if qgz > qrmin:
                no_fall = 0
            else:
                no_fall = 1

        with interval(1, None):

            if no_fall[0, 0, -1] == 1:

                if qgz > qrmin:
                    no_fall = 0
                else:
                    no_fall = 1

            else:

                no_fall = 0

    with computation(BACKWARD), interval(0, -1):

        if no_fall[0, 0, +1] == 0:
            no_fall = no_fall[0, 0, +1]

    with computation(PARALLEL), interval(...):

        if no_fall == 1:
            g1 = 0.0

    with computation(PARALLEL):

        with interval(1, -1):

            if no_fall == 0:
                zt = ze - dt5 * (vtgz[0, 0, -1] + vtgz)

        with interval(-1, None):

            if no_fall == 0:

                zt = ze - dt5 * (vtgz[0, 0, -1] + vtgz)
                zt_kbot1 = zs - dts * vtgz

    with computation(FORWARD):

        with interval(1, -1):

            if (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                zt = zt[0, 0, -1] - dz_min

        with interval(-1, None):

            if (no_fall[0, 0, -1] == 0) and (zt >= zt[0, 0, -1]):
                zt = zt[0, 0, -1] - dz_min

            if (no_fall == 0) and (zt_kbot1 >= zt):
                zt_kbot1 = zt - dz_min

    with computation(BACKWARD), interval(0, -1):

        if no_fall == 0:
            zt_kbot1 = zt_kbot1[0, 0, +1] - dz_min

    with computation(PARALLEL), interval(...):

        if no_fall == 0:

            if do_sedi_w == 1:
                dm = dp1 * (1.0 + qvz + qlz + qrz + qiz + qsz + qgz)

    with computation(PARALLEL):

        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - ze[0, 0, +1]

        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                dz = ze - zs

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            dd = dts * vtgz
            qgz = qgz * dp1

    # Sedimentation
    with computation(FORWARD):

        with interval(0, 1):

            if (use_ppm == 0) and (no_fall == 0):
                qm = qgz / (dz + dd)

        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                qm = (qgz[0, 0, 0] + dd[0, 0, -1] * qm[0, 0, -1]) / (dz + dd)

    with computation(PARALLEL), interval(...):

        if (use_ppm == 0) and (no_fall == 0):

            # qm is density at this stage
            qm = qm * dz

    # Output mass fluxes
    with computation(FORWARD):

        with interval(0, 1):

            if (use_ppm == 0) and (no_fall == 0):
                m1_tf = qgz - qm

        with interval(1, None):

            if (use_ppm == 0) and (no_fall == 0):
                m1_tf = m1_tf[0, 0, -1] + qgz[0, 0, 0] - qm

    with computation(BACKWARD):

        with interval(-1, None):

            if (use_ppm == 0) and (no_fall == 0):
                g1 = m1_tf

        with interval(0, -1):

            if (use_ppm == 0) and (no_fall == 0):
                g1 = g1[0, 0, +1]

    with computation(PARALLEL):

        with interval(0, 1):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qgz = qm / dp1

                m1_sol = m1_sol + m1_tf

                # Vertical velocity transportation during sedimentation
                if do_sedi_w == 1:
                    w = (dm * w + m1_tf * vtgz) / (dm - m1_tf)

        with interval(1, None):

            if no_fall == 0:

                if use_ppm == 0:

                    # Update
                    qgz = qm / dp1

                m1_sol = m1_sol + m1_tf

                if do_sedi_w == 1:

                    w[0, 0, 0] = (
                        dm * w[0, 0, 0]
                        - m1_tf[0, 0, -1] * vtgz[0, 0, -1]
                        + m1_tf * vtgz
                    ) / (dm + m1_tf[0, 0, -1] - m1_tf)

    with computation(PARALLEL), interval(...):

        rain = rain + r1  # From melted snow and ice that reached the ground
        snow = snow + s1
        graupel = graupel + g1
        ice = ice + i1

    # Heat transportation during sedimentation
    with computation(PARALLEL):

        with interval(0, 1):

            if do_sedi_heat == 1:

                # Input q fields are dry mixing ratios, and dm is dry air mass
                dgz = -0.5 * grav * dz1
                cvn = dp1 * (
                    cv_air
                    + qvz * cv_vap
                    + (qrz + qlz) * c_liq
                    + (qiz + qsz + qgz) * c_ice
                )

                # - Assumption: The ke in the falling condensates is negligible
                #               compared to the potential energy that was
                #               unaccounted for. Local thermal equilibrium is
                #               assumed, and the loss in pe is transformed into
                #               internal energy (to heat the whole grid box).
                # - Backward time-implicit upwind transport scheme:
                # - dm here is dry air mass
                tmp = cvn + m1_sol * c_ice
                tz = tz + m1_sol * dgz / tmp

        with interval(1, None):

            if do_sedi_heat == 1:

                # Input q fields are dry mixing ratios, and dm is dry air mass
                dgz = -0.5 * grav * dz1
                cvn = dp1 * (
                    cv_air
                    + qvz * cv_vap
                    + (qrz + qlz) * c_liq
                    + (qiz + qsz + qgz) * c_ice
                )

    # Implicit algorithm
    with computation(FORWARD), interval(1, None):

        if do_sedi_heat == 1:

            tz[0, 0, 0] = (
                (cvn + c_ice * (m1_sol - m1_sol[0, 0, -1])) * tz[0, 0, 0]
                + m1_sol[0, 0, -1] * c_ice * tz[0, 0, -1]
                + dgz * (m1_sol[0, 0, -1] + m1_sol)
            ) / (cvn + c_ice * m1_sol)


def icloud(
    h_var: FloatField,
    rh_adj: FloatField,
    rh_rain: FloatField,
    qaz: FloatField,
    qgz: FloatField,
    qiz: FloatField,
    qlz: FloatField,
    qrz: FloatField,
    qsz: FloatField,
    qvz: FloatField,
    tz: FloatField,
    den: FloatField,
    denfac: FloatField,
    p1: FloatField,
    vtgz: FloatField,
    vtrz: FloatField,
    vtsz: FloatField,
    c_air: Float,
    c_vap: Float,
    d0_vap: Float,
    lv00: Float,
    cracs: Float,
    csacr: Float,
    cgacr: Float,
    cgacs: Float,
    acco_00: Float,
    acco_01: Float,
    acco_02: Float,
    acco_03: Float,
    acco_10: Float,
    acco_11: Float,
    acco_12: Float,
    acco_13: Float,
    acco_20: Float,
    acco_21: Float,
    acco_22: Float,
    acco_23: Float,
    csacw: Float,
    csaci: Float,
    cgacw: Float,
    cgaci: Float,
    cracw: Float,
    cssub_0: Float,
    cssub_1: Float,
    cssub_2: Float,
    cssub_3: Float,
    cssub_4: Float,
    cgfr_0: Float,
    cgfr_1: Float,
    csmlt_0: Float,
    csmlt_1: Float,
    csmlt_2: Float,
    csmlt_3: Float,
    csmlt_4: Float,
    cgmlt_0: Float,
    cgmlt_1: Float,
    cgmlt_2: Float,
    cgmlt_3: Float,
    cgmlt_4: Float,
    ces0: Float,
    tice0: Float,
    t_wfr: Float,
    dts: Float,
    rdts: Float,
    fac_i2s: Float,
    fac_g2v: Float,
    fac_v2g: Float,
    fac_imlt: Float,
    fac_l2v: Float,
):

    with computation(PARALLEL), interval(...):

        # Ice-phase microphysics

        # Define heat capacity and latent heat coefficient
        lhi = li00 + dc_ice * tz
        q_liq = qlz + qrz
        q_sol = qiz + qsz + qgz
        cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
        icpk = lhi / cvm

        # - Sources of cloud ice: pihom, cold rain, and the sat_adj
        # - Sources of snow: cold rain, auto conversion + accretion (from cloud ice)
        # - sat_adj (deposition; requires pre-existing snow); initial snow comes from autoconversion

        t_wfr_tmp = t_wfr
        if (tz > tice) and (qiz > qcmin):

            # pimlt: instant melting of cloud ice
            melt = min(qiz, fac_imlt * (tz - tice) / icpk)
            tmp = min(melt, dim(ql_mlt, qlz))  # Maximum ql amount
            qlz = qlz + tmp
            qrz = qrz + melt - tmp
            qiz = qiz - melt
            q_liq = q_liq + melt
            q_sol = q_sol - melt
            cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
            tz = tz - melt * lhi / cvm

        elif (tz < t_wfr) and (qlz > qcmin):

            # - pihom: homogeneous freezing of cloud water into cloud ice
            # - This is the 1st occurance of liquid water freezing in the split mp process

            dtmp = t_wfr_tmp - tz
            factor = min(1.0, dtmp / dt_fr)
            sink = min(qlz * factor, dtmp / icpk)
            qi_crt = qi_gen * min(qi_lim, 0.1 * (tice - tz)) / den
            tmp = min(sink, dim(qi_crt, qiz))
            qlz = qlz - sink
            qsz = qsz + sink - tmp
            qiz = qiz + tmp
            q_liq = q_liq - sink
            q_sol = q_sol + sink
            cvm = c_air + qvz * c_vap + q_liq * c_liq + q_sol * c_ice
            tz = tz + sink * lhi / cvm

    # Vertical subgrid variability
    with computation(FORWARD):

        with interval(0, 1):

            if z_slope_ice == 1:
                di = 0.0

        with interval(1, None):

            if z_slope_ice == 1:
                dq = 0.5 * (qiz[0, 0, 0] - qiz[0, 0, -1])

    with computation(PARALLEL):

        with interval(1, -1):

            if z_slope_ice == 1:

                # Use twice the strength of the positive definiteness limiter (lin et al 1994)
                di = 0.5 * min(abs(dq + dq[0, 0, +1]), 0.5 * qiz[0, 0, 0])

                if dq * dq[0, 0, +1] <= 0.0:

                    if dq > 0.0:  # Local maximum

                        di = min(di, min(dq, -dq[0, 0, +1]))

                    else:

                        di = 0.0

        with interval(-1, None):

            if z_slope_ice == 1:
                di = 0.0

    with computation(PARALLEL), interval(...):

        if z_slope_ice == 1:

            # Impose a presumed background horizontal variability that is
            # proportional to the value itself
            di = max(di, max(qvmin, h_var * qiz))

        else:

            di = max(qvmin, h_var * qiz)

        qaz, qgz, qiz, qlz, qrz, qsz, qvz, tz = icloud_main(
            c_air,
            c_vap,
            d0_vap,
            lv00,
            cracs,
            csacr,
            cgacr,
            cgacs,
            acco_00,
            acco_01,
            acco_02,
            acco_03,
            acco_10,
            acco_11,
            acco_12,
            acco_13,
            acco_20,
            acco_21,
            acco_22,
            acco_23,
            csacw,
            csaci,
            cgacw,
            cgaci,
            cssub_0,
            cssub_1,
            cssub_2,
            cssub_3,
            cssub_4,
            cgfr_0,
            cgfr_1,
            csmlt_0,
            csmlt_1,
            csmlt_2,
            csmlt_3,
            csmlt_4,
            cgmlt_0,
            cgmlt_1,
            cgmlt_2,
            cgmlt_3,
            cgmlt_4,
            ces0,
            tice0,
            t_wfr,
            dts,
            rdts,
            fac_i2s,
            fac_g2v,
            fac_v2g,
            fac_l2v,
            h_var,
            rh_adj,
            rh_rain,
            qaz,
            qgz,
            qiz,
            qlz,
            qrz,
            qsz,
            qvz,
            tz,
            den,
            denfac,
            vtgz,
            vtrz,
            vtsz,
            p1,
            di,
            q_liq,
            q_sol,
            cvm,
        )


def fields_update(
    graupel: FloatField,
    ice: FloatField,
    rain: FloatField,
    snow: FloatField,
    qaz: FloatField,
    qgz: FloatField,
    qiz: FloatField,
    qlz: FloatField,
    qrz: FloatField,
    qsz: FloatField,
    qvz: FloatField,
    tz: FloatField,
    udt: FloatField,
    vdt: FloatField,
    qa_dt: FloatField,
    qg_dt: FloatField,
    qi_dt: FloatField,
    ql_dt: FloatField,
    qr_dt: FloatField,
    qs_dt: FloatField,
    qv_dt: FloatField,
    pt_dt: FloatField,
    qa0: FloatField,
    qg0: FloatField,
    qi0: FloatField,
    ql0: FloatField,
    qr0: FloatField,
    qs0: FloatField,
    qv0: FloatField,
    t0: FloatField,
    dp0: FloatField,
    u0: FloatField,
    v0: FloatField,
    dp1: FloatField,
    u1: FloatField,
    v1: FloatField,
    m1: FloatField,
    m2_rain: FloatField,
    m2_sol: FloatField,
    ntimes: Int,
    c_air: Float,
    c_vap: Float,
    rdt: Float,
):

    with computation(PARALLEL), interval(...):

        # Convert units from Pa*kg/kg to kg/m^2/s
        m2_rain = m2_rain * rdt * rgrav
        m2_sol = m2_sol * rdt * rgrav

    # Momentum transportation during sedimentation (dp1 is dry mass; dp0
    # is the old moist total mass)
    with computation(FORWARD), interval(1, None):

        if sedi_transport == 1:

            u1[0, 0, 0] = (dp0[0, 0, 0] * u1[0, 0, 0] + m1[0, 0, -1] * u1[0, 0, -1]) / (
                dp0[0, 0, 0] + m1[0, 0, -1]
            )
            v1[0, 0, 0] = (dp0[0, 0, 0] * v1[0, 0, 0] + m1[0, 0, -1] * v1[0, 0, -1]) / (
                dp0[0, 0, 0] + m1[0, 0, -1]
            )

    with computation(PARALLEL), interval(1, None):

        if sedi_transport == 1:

            udt = udt + (u1 - u0) * rdt
            vdt = vdt + (v1 - v0) * rdt

    with computation(PARALLEL), interval(...):

        # Update moist air mass (actually hydrostatic pressure) and convert
        # to dry mixing ratios
        omq = dp1 / dp0
        qv_dt = qv_dt + rdt * (qvz - qv0) * omq
        ql_dt = ql_dt + rdt * (qlz - ql0) * omq
        qr_dt = qr_dt + rdt * (qrz - qr0) * omq
        qi_dt = qi_dt + rdt * (qiz - qi0) * omq
        qs_dt = qs_dt + rdt * (qsz - qs0) * omq
        qg_dt = qg_dt + rdt * (qgz - qg0) * omq

        cvm = c_air + qvz * c_vap + (qrz + qlz) * c_liq + (qiz + qsz + qgz) * c_ice

        pt_dt = pt_dt + rdt * (tz - t0) * cvm / cp_air

        # Update cloud fraction tendency
        if do_qa == 1:

            qa_dt = 0.0

        else:

            qa_dt = qa_dt + rdt * (qaz / ntimes - qa0)

        """
        LEFT OUT FOR NOW
        # No clouds allowed above ktop
        if k_s < ktop:
            qa_dt[:, :, k_s:ktop+1] = 0.
        """

        # Convert to mm / day
        convt = 86400.0 * rdt * rgrav

        rain = rain * convt
        snow = snow * convt
        ice = ice * convt
        graupel = graupel * convt


class MicrophysicsState:
    """
    pt, qvapor, qrain, qice, qsnow, qgraupel, qcld, &
    ua, va, delp, delz, omga: same as physics state
    qv_dt: specific humidity tendency
    ql_dt: cloud water mixing ratio tendency
    qr_dt: rain water mixing ratio tendency
    qi_dt: ice water mixing ratio tendency
    qs_dt: snow mixing ratio tendency
    qg_dt: graupel mixing ratio tendency
    qa_dt: cloud fraction tendency
    udt: eastard wind tendency
    vdt: northward wind tendency
    pt_dt: air temperature tendency
    """

    def __init__(
        self,
        pt: FloatField,
        qvapor: FloatField,
        qliquid: FloatField,
        qrain: FloatField,
        qice: FloatField,
        qsnow: FloatField,
        qgraupel: FloatField,
        qcld: FloatField,
        ua: FloatField,
        va: FloatField,
        delp: FloatField,
        delz: FloatField,
        omga: FloatField,
        delprsi: FloatField,
        wmp: FloatField,
        dz: FloatField,
        tendency_storage: FloatField,
    ):
        self.pt = pt
        self.qvapor = qvapor
        self.qliquid = qliquid
        self.qrain = qrain
        self.qice = qice
        self.qsnow = qsnow
        self.qgraupel = qgraupel
        self.qcld = qcld
        self.ua = ua
        self.va = va
        self.delp = delp
        self.delz = delz
        self.omga = omga
        self.qv_dt = copy.deepcopy(tendency_storage)
        self.ql_dt = copy.deepcopy(tendency_storage)
        self.qr_dt = copy.deepcopy(tendency_storage)
        self.qi_dt = copy.deepcopy(tendency_storage)
        self.qs_dt = copy.deepcopy(tendency_storage)
        self.qg_dt = copy.deepcopy(tendency_storage)
        self.qa_dt = copy.deepcopy(tendency_storage)
        self.udt = copy.deepcopy(tendency_storage)
        self.vdt = copy.deepcopy(tendency_storage)
        self.pt_dt = copy.deepcopy(tendency_storage)
        self.delprsi = delprsi
        self.wmp = wmp
        self.dz = dz


class Microphysics:
    def __init__(self, stencil_factory: StencilFactory, grid, namelist):
        # [TODO]: many of the "constants" come from namelist, needs to be updated
        self.gfdl_cloud_microphys_init()
        self.grid = grid
        self.namelist = namelist
        origin = self.grid.compute_origin()
        shape = self.grid.domain_shape_full(add=(1, 1, 1))

        self._hydrostatic = self.namelist.hydrostatic
        self._kke = self.grid.npz - 1
        self._kbot = self.grid.npz - 1
        self._k_s = 0
        self._k_e = self._kke - self._k_s + 1
        self._dt_atmos = self.namelist.dt_atmos
        # Define heat capacity of dry air and water vapor based on
        # hydrostatical property, [TODO] (EW): investigate why this is hard coded
        self._c_air = cp_air
        self._c_vap = cp_vap
        self._p_nonhydro = 0
        self._d0_vap = self._c_vap - c_liq
        self._lv00 = hlv0 - self._d0_vap * t_ice
        self._do_sedi_w = 0 if self._hydrostatic else 1
        # Define cloud microphysics sub time step
        self._mpdt = min(self._dt_atmos, mp_time)
        self._rdt = 1.0 / self._dt_atmos
        self._ntimes = Int(round(self._dt_atmos / self._mpdt))
        # Small time step
        self._dts = self._dt_atmos / self._ntimes
        self._dt_rain = self._dts * 0.5
        # Calculate cloud condensation nuclei (ccn) based on klein eq. 15
        self._cpaut = c_paut * 0.104 * grav / 1.717e-5
        self._use_ccn = 0 if prog_ccn == 1 else 1
        self._area = self.grid.area
        self._land = utils.make_storage_from_shape(shape, origin=origin, init=True)
        self._rain = utils.make_storage_from_shape(shape, origin=origin, init=True)
        self._graupel = utils.make_storage_from_shape(shape, origin=origin, init=True)
        self._ice = utils.make_storage_from_shape(shape, origin=origin, init=True)
        self._snow = utils.make_storage_from_shape(shape, origin=origin, init=True)

        self._h_var = utils.make_storage_from_shape(shape, origin=origin)
        self._rh_adj = utils.make_storage_from_shape(shape, origin=origin)
        self._rh_rain = utils.make_storage_from_shape(shape, origin=origin)

        self._qn = utils.make_storage_from_shape(shape, origin=origin, init=True)
        self._qaz = utils.make_storage_from_shape(shape, origin=origin, init=True)
        self._qgz = utils.make_storage_from_shape(shape, origin=origin)
        self._qiz = utils.make_storage_from_shape(shape, origin=origin)
        self._qlz = utils.make_storage_from_shape(shape, origin=origin)
        self._qrz = utils.make_storage_from_shape(shape, origin=origin)
        self._qsz = utils.make_storage_from_shape(shape, origin=origin)
        self._qvz = utils.make_storage_from_shape(shape, origin=origin)
        self._den = utils.make_storage_from_shape(shape, origin=origin)
        self._denfac = utils.make_storage_from_shape(shape, origin=origin)
        self._tz = utils.make_storage_from_shape(shape, origin=origin)
        self._qa0 = utils.make_storage_from_shape(shape, origin=origin)
        self._qg0 = utils.make_storage_from_shape(shape, origin=origin)
        self._qi0 = utils.make_storage_from_shape(shape, origin=origin)
        self._ql0 = utils.make_storage_from_shape(shape, origin=origin)
        self._qr0 = utils.make_storage_from_shape(shape, origin=origin)
        self._qs0 = utils.make_storage_from_shape(shape, origin=origin)
        self._qv0 = utils.make_storage_from_shape(shape, origin=origin)
        self._t0 = utils.make_storage_from_shape(shape, origin=origin)
        self._dp0 = utils.make_storage_from_shape(shape, origin=origin)
        self._den0 = utils.make_storage_from_shape(shape, origin=origin)
        self._dz0 = utils.make_storage_from_shape(shape, origin=origin)
        self._u0 = utils.make_storage_from_shape(shape, origin=origin)
        self._v0 = utils.make_storage_from_shape(shape, origin=origin)
        self._dz1 = utils.make_storage_from_shape(shape, origin=origin)
        self._dp1 = utils.make_storage_from_shape(shape, origin=origin)
        self._p1 = utils.make_storage_from_shape(shape, origin=origin)
        self._u1 = utils.make_storage_from_shape(shape, origin=origin)
        self._v1 = utils.make_storage_from_shape(shape, origin=origin)
        self._m1 = utils.make_storage_from_shape(shape, origin=origin, init=True)
        self._vtgz = utils.make_storage_from_shape(shape, origin=origin)
        self._vtrz = utils.make_storage_from_shape(shape, origin=origin)
        self._vtsz = utils.make_storage_from_shape(shape, origin=origin)
        self._ccn = utils.make_storage_from_shape(shape, origin=origin)
        self._c_praut = utils.make_storage_from_shape(shape, origin=origin)
        self._m1_sol = utils.make_storage_from_shape(shape, origin=origin)
        self._m2_rain = utils.make_storage_from_shape(shape, origin=origin, init=True)
        self._m2_sol = utils.make_storage_from_shape(shape, origin=origin, init=True)

        self._so3 = 7.0 / 3.0
        self._zs = 0.0
        self._rdts = 1.0 / self._dts
        self._dt_evap = 0.5 * self._dts if fast_sat_adj else self._dts
        self._fac_i2s = 1.0 - np.exp(-self._dts / tau_i2s)
        self._fac_g2v = 1.0 - np.exp(-self._dts / tau_g2v)
        self._fac_v2g = 1.0 - np.exp(-self._dts / tau_v2g)
        self._fac_imlt = 1.0 - np.exp(-0.5 * self._dts / tau_imlt)
        self._fac_l2v = 1.0 - np.exp(-self._dt_evap / tau_l2v)

        self._fields_init = stencil_factory.from_origin_domain(
            func=fields_init,
            origin=self.grid.grid_indexing.origin_compute(),
            domain=self.grid.grid_indexing.domain_compute(),
        )
        self._warm_rain = stencil_factory.from_origin_domain(
            func=warm_rain,
            origin=self.grid.grid_indexing.origin_compute(),
            domain=self.grid.grid_indexing.domain_compute(),
        )
        self._sedimentation = stencil_factory.from_origin_domain(
            func=sedimentation,
            origin=self.grid.grid_indexing.origin_compute(),
            domain=self.grid.grid_indexing.domain_compute(),
        )
        self._icloud = stencil_factory.from_origin_domain(
            func=icloud,
            origin=self.grid.grid_indexing.origin_compute(),
            domain=self.grid.grid_indexing.domain_compute(),
        )
        self._fields_update = stencil_factory.from_origin_domain(
            func=fields_update,
            origin=self.grid.grid_indexing.origin_compute(),
            domain=self.grid.grid_indexing.domain_compute(),
        )

    def gfdl_cloud_microphys_init(self):
        self.setupm()
        self._log_10 = np.log(10.0)
        self._tice0 = tice - 0.01
        self._t_wfr = tice - 40.0

    def setupm(self):
        gam263 = 1.456943
        gam275 = 1.608355
        gam290 = 1.827363
        gam325 = 2.54925
        gam350 = 3.323363
        gam380 = 4.694155

        # Intercept parameters
        rnzs = 3.0e6
        rnzr = 8.0e6
        rnzg = 4.0e6

        # Density parameters
        acc = np.array([5.0, 2.0, 0.5])

        pie = 4.0 * np.arctan(1.0)

        # S. Klein's formular (eq 16) from am2
        fac_rc = (4.0 / 3.0) * pie * rhor * rthresh ** 3

        vdifu = 2.11e-5
        tcond = 2.36e-2

        visk = 1.259e-5
        hlts = 2.8336e6
        hltc = 2.5e6
        hltf = 3.336e5

        ch2o = 4.1855e3

        pisq = pie * pie
        scm3 = (visk / vdifu) ** (1.0 / 3.0)

        cracs = pisq * rnzr * rnzs * rhos
        csacr = pisq * rnzr * rnzs * rhor
        cgacr = pisq * rnzr * rnzg * rhor
        cgacs = pisq * rnzg * rnzs * rhos
        cgacs = cgacs * c_pgacs

        act = np.empty(8)
        act[0] = pie * rnzs * rhos
        act[1] = pie * rnzr * rhor
        act[5] = pie * rnzg * rhog
        act[2] = act[1]
        act[3] = act[0]
        act[4] = act[1]
        act[6] = act[0]
        act[7] = act[5]

        acco = np.empty((3, 4))
        for i in range(3):
            for k in range(4):
                acco[i, k] = acc[i] / (
                    act[2 * k] ** ((6 - i) * 0.25) * act[2 * k + 1] ** ((i + 1) * 0.25)
                )

        gcon = 40.74 * np.sqrt(sfcrho)

        # Decreasing csacw to reduce cloud water --- > snow
        csacw = pie * rnzs * clin * gam325 / (4.0 * act[0] ** 0.8125)

        craci = pie * rnzr * alin * gam380 / (4.0 * act[1] ** 0.95)
        csaci = csacw * c_psaci

        cgacw = pie * rnzg * gam350 * gcon / (4.0 * act[5] ** 0.875)

        cgaci = cgacw * 0.05

        cracw = craci
        cracw = c_cracw * cracw

        # Subl and revap: five constants for three separate processes
        cssub = np.empty(5)
        cssub[0] = 2.0 * pie * vdifu * tcond * rvgas * rnzs
        cssub[1] = 0.78 / np.sqrt(act[0])
        cssub[2] = 0.31 * scm3 * gam263 * np.sqrt(clin / visk) / act[0] ** 0.65625
        cssub[3] = tcond * rvgas
        cssub[4] = (hlts ** 2) * vdifu

        cgsub = np.empty(5)
        cgsub[0] = 2.0 * pie * vdifu * tcond * rvgas * rnzg
        cgsub[1] = 0.78 / np.sqrt(act[5])
        cgsub[2] = 0.31 * scm3 * gam275 * np.sqrt(gcon / visk) / act[5] ** 0.6875
        cgsub[3] = cssub[3]
        cgsub[4] = cssub[4]

        crevp = np.empty(5)
        crevp[0] = 2.0 * pie * vdifu * tcond * rvgas * rnzr
        crevp[1] = 0.78 / np.sqrt(act[1])
        crevp[2] = 0.31 * scm3 * gam290 * np.sqrt(alin / visk) / act[1] ** 0.725
        crevp[3] = cssub[3]
        crevp[4] = hltc ** 2 * vdifu

        cgfr = np.empty(2)
        cgfr[0] = 20.0e2 * pisq * rnzr * rhor / act[1] ** 1.75
        cgfr[1] = 0.66

        # smlt: five constants (lin et al. 1983)
        csmlt = np.empty(5)
        csmlt[0] = 2.0 * pie * tcond * rnzs / hltf
        csmlt[1] = 2.0 * pie * vdifu * rnzs * hltc / hltf
        csmlt[2] = cssub[1]
        csmlt[3] = cssub[2]
        csmlt[4] = ch2o / hltf

        # gmlt: five constants
        cgmlt = np.empty(5)
        cgmlt[0] = 2.0 * pie * tcond * rnzg / hltf
        cgmlt[1] = 2.0 * pie * vdifu * rnzg * hltc / hltf
        cgmlt[2] = cgsub[1]
        cgmlt[3] = cgsub[2]
        cgmlt[4] = ch2o / hltf

        es0 = 6.107799961e2  # ~6.1 mb
        self._fac_rc = fac_rc

        self._cracs = cracs
        self._csacr = csacr
        self._cgacr = cgacr
        self._cgacs = cgacs
        self._acco = acco
        self._csacw = csacw
        self._csaci = csaci
        self._cgacw = cgacw
        self._cgaci = cgaci
        self._cracw = cracw
        self._cssub = cssub
        self._crevp = crevp
        self._cgfr = cgfr
        self._csmlt = csmlt
        self._cgmlt = cgmlt
        self._ces0 = eps * es0

    def __call__(self, state: MicrophysicsState):
        self._fields_init(
            self._land,
            self._area,
            self._h_var,
            self._rh_adj,
            self._rh_rain,
            self._graupel,
            self._ice,
            self._rain,
            self._snow,
            state.qcld,
            state.qgraupel,
            state.qice,
            state.qliquid,
            self._qn,
            state.qrain,
            state.qsnow,
            state.qvapor,
            state.pt,
            state.delprsi,
            state.delz,
            self._qgz,
            self._qiz,
            self._qlz,
            self._qrz,
            self._qsz,
            self._qvz,
            self._tz,
            state.qi_dt,
            state.qs_dt,
            state.ua,
            state.va,
            self._qa0,
            self._qg0,
            self._qi0,
            self._ql0,
            self._qr0,
            self._qs0,
            self._qv0,
            self._t0,
            self._dp0,
            self._den0,
            self._dz0,
            self._u0,
            self._v0,
            self._dp1,
            self._p1,
            self._u1,
            self._v1,
            self._ccn,
            self._c_praut,
            Int(self._use_ccn),
            self._c_air,
            self._c_vap,
            self._d0_vap,
            self._lv00,
            Float(self._dt_atmos),
            self._rdt,
            self._cpaut,
        )

        for n in range(self._ntimes):
            self._warm_rain(
                self._h_var,
                self._rain,
                self._qgz,
                self._qiz,
                self._qlz,
                self._qrz,
                self._qsz,
                self._qvz,
                self._tz,
                self._den,
                self._denfac,
                state.wmp,
                self._t0,
                self._den0,
                self._dz0,
                self._dz1,
                self._dp1,
                self._m1,
                self._vtrz,
                self._ccn,
                self._c_praut,
                self._m1_sol,
                self._m2_rain,
                self._m2_sol,
                Int(1),
                Int(self._do_sedi_w),
                Int(self._p_nonhydro),
                Int(self._use_ccn),
                self._c_air,
                self._c_vap,
                self._d0_vap,
                self._lv00,
                self._fac_rc,
                self._cracw,
                self._crevp[0],
                self._crevp[1],
                self._crevp[2],
                self._crevp[3],
                self._crevp[4],
                self._t_wfr,
                self._so3,
                self._dt_rain,
                self._zs,
            )
            self._sedimentation(
                self._graupel,
                self._ice,
                self._rain,
                self._snow,
                self._qgz,
                self._qiz,
                self._qlz,
                self._qrz,
                self._qsz,
                self._qvz,
                self._tz,
                self._den,
                state.wmp,
                self._dz1,
                self._dp1,
                self._vtgz,
                self._vtsz,
                self._m1_sol,
                Int(self._do_sedi_w),
                self._c_air,
                self._c_vap,
                self._d0_vap,
                self._lv00,
                self._log_10,
                self._zs,
                self._dts,
                self._fac_imlt,
            )
            self._warm_rain(
                self._h_var,
                self._rain,
                self._qgz,
                self._qiz,
                self._qlz,
                self._qrz,
                self._qsz,
                self._qvz,
                self._tz,
                self._den,
                self._denfac,
                state.wmp,
                self._t0,
                self._den0,
                self._dz0,
                self._dz1,
                self._dp1,
                self._m1,
                self._vtrz,
                self._ccn,
                self._c_praut,
                self._m1_sol,
                self._m2_rain,
                self._m2_sol,
                Int(0),
                Int(self._do_sedi_w),
                Int(self._p_nonhydro),
                Int(self._use_ccn),
                self._c_air,
                self._c_vap,
                self._d0_vap,
                self._lv00,
                self._fac_rc,
                self._cracw,
                self._crevp[0],
                self._crevp[1],
                self._crevp[2],
                self._crevp[3],
                self._crevp[4],
                self._t_wfr,
                self._so3,
                self._dt_rain,
                self._zs,
            )
            self._icloud(
                self._h_var,
                self._rh_adj,
                self._rh_rain,
                self._qaz,
                self._qgz,
                self._qiz,
                self._qlz,
                self._qrz,
                self._qsz,
                self._qvz,
                self._tz,
                self._den,
                self._denfac,
                self._p1,
                self._vtgz,
                self._vtrz,
                self._vtsz,
                self._c_air,
                self._c_vap,
                self._d0_vap,
                self._lv00,
                self._cracs,
                self._csacr,
                self._cgacr,
                self._cgacs,
                self._acco[0, 0],
                self._acco[0, 1],
                self._acco[0, 2],
                self._acco[0, 3],
                self._acco[1, 0],
                self._acco[1, 1],
                self._acco[1, 2],
                self._acco[1, 3],
                self._acco[2, 0],
                self._acco[2, 1],
                self._acco[2, 2],
                self._acco[2, 3],
                self._csacw,
                self._csaci,
                self._cgacw,
                self._cgaci,
                self._cracw,
                self._cssub[0],
                self._cssub[1],
                self._cssub[2],
                self._cssub[3],
                self._cssub[4],
                self._cgfr[0],
                self._cgfr[1],
                self._csmlt[0],
                self._csmlt[1],
                self._csmlt[2],
                self._csmlt[3],
                self._csmlt[4],
                self._cgmlt[0],
                self._cgmlt[1],
                self._cgmlt[2],
                self._cgmlt[3],
                self._cgmlt[4],
                self._ces0,
                self._tice0,
                self._t_wfr,
                self._dts,
                self._rdts,
                self._fac_i2s,
                self._fac_g2v,
                self._fac_v2g,
                self._fac_imlt,
                self._fac_l2v,
            )
        self._fields_update(
            self._graupel,
            self._ice,
            self._rain,
            self._snow,
            self._qaz,
            self._qgz,
            self._qiz,
            self._qlz,
            self._qrz,
            self._qsz,
            self._qvz,
            self._tz,
            state.udt,
            state.vdt,
            state.qa_dt,
            state.qg_dt,
            state.qi_dt,
            state.ql_dt,
            state.qr_dt,
            state.qs_dt,
            state.qv_dt,
            state.pt_dt,
            self._qa0,
            self._qg0,
            self._qi0,
            self._ql0,
            self._qr0,
            self._qs0,
            self._qv0,
            self._t0,
            self._dp0,
            self._u0,
            self._v0,
            self._dp1,
            self._u1,
            self._v1,
            self._m1,
            self._m2_rain,
            self._m2_sol,
            self._ntimes,
            self._c_air,
            self._c_vap,
            self._rdt,
        )
        print("Microphysics")
