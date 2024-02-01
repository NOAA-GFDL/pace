import gt4py.cartesian.gtscript as gtscript
from gt4py.cartesian.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    exp,
    interval,
    log,
)

import ndsl.constants as constants
from ndsl.dsl.typing import Float, FloatField


@gtscript.function
def set_cappa(qvapor, cvm, r_vir):
    cappa = constants.RDGAS / (constants.RDGAS + cvm / (1.0 + r_vir * qvapor))
    return cappa


@gtscript.function
def moist_cvm(qvapor, gz, ql, qs):
    cvm = (
        (1.0 - (qvapor + gz)) * constants.CV_AIR
        + qvapor * constants.CV_VAP
        + ql * constants.C_LIQ
        + qs * constants.C_ICE
    )
    return cvm


@gtscript.function
def moist_cv_nwat6_fn(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
):
    ql = qliquid + qrain
    qs = qice + qsnow + qgraupel
    gz = ql + qs
    cvm = moist_cvm(qvapor, gz, ql, qs)
    return cvm, gz


@gtscript.function
def moist_pt_func(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    q_con: FloatField,
    pt: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    r_vir: Float,
):
    cvm, gz = moist_cv_nwat6_fn(
        qvapor, qliquid, qrain, qsnow, qice, qgraupel
    )  # if (nwat == 6) else moist_cv_default_fn(constants.CV_AIR)
    q_con = gz
    cappa = set_cappa(qvapor, cvm, r_vir)
    pt = pt * exp(cappa / (1.0 - cappa) * log(constants.RDG * delp / delz * pt))
    return cvm, gz, q_con, cappa, pt


@gtscript.function
def last_pt(
    pt: FloatField,
    dtmp: Float,
    pkz: FloatField,
    gz: FloatField,
    qv: FloatField,
    zvir: Float,
):
    return (pt + dtmp * pkz) / ((1.0 + zvir * qv) * (1.0 - gz))


def moist_pt_last_step(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    gz: FloatField,
    pt: FloatField,
    pkz: FloatField,
    dtmp: Float,
    r_vir: Float,
):
    """
    Args:
        qvapor (in):
        qliquid (in):
        qrain (in):
        qsnow (in):
        qice (in):
        qgraupel (in):
        gz (out):
        pt (inout):
        pkz (in):
        dtmp (in):
        r_vir (in):
    """
    with computation(PARALLEL), interval(...):
        # if nwat == 2:
        #    gz = qliquid if qliquid > 0. else 0.
        #    qv = qvapor if qvapor > 0. else 0.
        #    pt = last_pt(pt, dtmp, pkz, gz, qv, r_vir)
        # elif nwat == 6:
        gz = qliquid + qrain + qice + qsnow + qgraupel
        pt = last_pt(pt, dtmp, pkz, gz, qvapor, r_vir)
        # else:
        #    cvm, gz = moist_cv_nwat6_fn(qvapor, qliquid, qrain, qsnow, qice, qgraupel)
        #    pt = last_pt(pt, dtmp, pkz, gz, qvapor, zvir)


@gtscript.function
def compute_pkz_func(delp, delz, pt, cappa):
    # TODO use the exponential form for closer answer matching
    return exp(cappa * log(constants.RDG * delp / delz * pt))


def moist_pkz(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    q_con: FloatField,
    gz: FloatField,
    cvm: FloatField,
    pkz: FloatField,
    pt: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    r_vir: Float,
):
    """
    Args:
        qvapor (in):
        qliquid (in):
        qrain (in):
        qsnow (in):
        qice (in):
        qgraupel (in):
        q_con (out):
        gz (out):
        cvm (out):
        pkz (out):
        pt (in):
        cappa (out):
        delp (in):
        delz (in):
        r_vir (in):
    """
    # TODO: What is happening with q_con and gz here?
    with computation(PARALLEL), interval(...):
        cvm, gz = moist_cv_nwat6_fn(
            qvapor, qliquid, qrain, qsnow, qice, qgraupel
        )  # if (nwat == 6) else moist_cv_default_fn(constants.CV_AIR)
        q_con[0, 0, 0] = gz
        cappa = set_cappa(qvapor, cvm, r_vir)
        pkz = compute_pkz_func(delp, delz, pt, cappa)


def fv_setup(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    q_con: FloatField,
    cvm: FloatField,
    pkz: FloatField,
    pt: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    dp1: FloatField,
):
    """
    Args:
        qvapor (in):
        qliquid (in):
        qrain (in):
        qsnow (in):
        qice (in):
        qgraupel (in):
        q_con (out):
        cvm (out):
        pkz (out): p^(cappa)
        pt (in):
        cappa (out): Rd / Cp
        delp (in):
        delz (in):
        dp1 (out):
    """
    # without moist_cappa, we use a constant heat capacity for everything
    # variable heat capacity takes into account the mixing ratios of condensates
    # this is more accurate

    # TODO: what is being set up here, and how? update docstring
    with computation(PARALLEL), interval(...):
        from __externals__ import moist_phys

        if __INLINED(moist_phys):
            cvm, q_con = moist_cv_nwat6_fn(
                qvapor, qliquid, qrain, qsnow, qice, qgraupel
            )  # if (nwat == 6) else moist_cv_default_fn(constants.CV_AIR)
            dp1 = constants.ZVIR * qvapor
            cappa = constants.RDGAS / (constants.RDGAS + cvm / (1.0 + dp1))
            pkz = exp(
                cappa
                * log(constants.RDG * delp * pt * (1.0 + dp1) * (1.0 - q_con) / delz)
            )
            # TODO: find documentation reference
            # (1.0 + dp1) * (1.0 - q_con) takes out condensate mass,
            # described in more detail in fv3 docs
            # condensates don't obey ideal gas law so they must be taken out
        else:
            dp1 = 0
            pkz = exp(constants.KAPPA * log(constants.RDG * delp * pt / delz))
            # cell mean pressure based on ideal gas law, raised to cappa
            # exponential log structure is faster on most processors
