import numpy as np

import pace.fv3core.initialization.init_utils as init_utils
import pace.util as fv3util

from pace.fv3core.dycore_state import DycoreState
from pace.util.grid import GridData


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

    ps_output = init_utils._initialize_vortex_ps_phis(
        grid_data, shape, tc_properties, calc
    )
    ps, ps_u, ps_v = ps_output["ps"], ps_output["ps_uc"], ps_output["ps_vc"]

    # TODO restart file had different ak, bk. Figure out where they came from;
    # for now, take from metric terms
    ak = init_utils._define_ak()
    bk = init_utils._define_bk()
    delp = init_utils._initialize_delp(ak, bk, ps, shape)
    pe = init_utils._initialize_edge_pressure(delp, tc_properties["ptop"], shape)
    peln = np.log(pe)
    pk, pkz = init_utils.initialize_kappa_pressures(pe, peln, tc_properties["ptop"])

    pe_u = init_utils._initialize_edge_pressure_cgrid(
        ak, bk, ps_u, shape, tc_properties["ptop"]
    )
    pe_v = init_utils._initialize_edge_pressure_cgrid(
        ak, bk, ps_v, shape, tc_properties["ptop"]
    )

    ud, vd = init_utils._initialize_wind_dgrid(
        grid_data, tc_properties, calc, pe_u, pe_v, ps_u, ps_v, shape
    )
    ua, va = init_utils._interpolate_winds_dgrid_agrid(
        grid_data, ud, vd, tc_properties, shape
    )

    qvapor, pt = init_utils._initialize_qvapor_temperature(
        grid_data, pe, ps, tc_properties, calc, shape
    )
    delz, w = init_utils._initialize_delz_w(
        pe, ps, pt, qvapor, tc_properties, calc, shape
    )

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
