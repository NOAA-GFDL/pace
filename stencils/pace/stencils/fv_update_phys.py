import gt4py.gtscript as gtscript
from gt4py.gtscript import FORWARD, PARALLEL, computation, exp, interval, log

import pace.dsl.gt4py_utils as utils
import pace.util
import pace.util.constants as constants
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import Float, FloatField, FloatFieldIJ
from pace.stencils.c2l_ord import CubedToLatLon
from pace.stencils.update_dwind_phys import AGrid2DGridPhysics
from pace.util.grid import DriverGridData, GridData


# TODO: This is the same as moist_cv.py in fv3core, should move to integration dir
@gtscript.function
def moist_cvm(qvapor, gz, ql, qs):
    cvm = (
        (1.0 - (qvapor + gz)) * constants.CV_AIR
        + qvapor * constants.CV_VAP
        + ql * constants.C_LIQ
        + qs * constants.C_ICE
    )
    return cvm


# This is based off of moist_cv_nwat6_fn gt4py function in moist_cv.py in fv3core
def moist_cv(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    pt: FloatField,
    t_dt: FloatField,
    con_cp: Float,
    dt: Float,
):
    with computation(PARALLEL), interval(...):
        ql = qliquid + qrain
        qs = qice + qsnow + qgraupel
        gz = ql + qs
        cvm = moist_cvm(qvapor, gz, ql, qs)
        pt = pt + t_dt * dt * con_cp / cvm


def update_pressure_and_surface_winds(
    pe: FloatField,
    delp: FloatField,
    peln: FloatField,
    pk: FloatField,
    KAPPA: Float,
    ua: FloatField,
    va: FloatField,
    ps: FloatFieldIJ,
    u_srf: FloatFieldIJ,
    v_srf: FloatFieldIJ,
):

    with computation(FORWARD), interval(1, None):
        pe = pe[0, 0, -1] + delp[0, 0, -1]
    with computation(PARALLEL), interval(1, None):
        peln = log(pe[0, 0, 0])
        pk = exp(KAPPA * peln[0, 0, 0])
    with computation(FORWARD), interval(-1, None):
        ps = pe
    with computation(FORWARD), interval(-2, -1):
        u_srf = ua[0, 0, 0]
        v_srf = va[0, 0, 0]


class ApplyPhysics2Dycore:
    """
    Fortran name is fv_update_phys
    Apply the physics tendencies (u_dt, v_dt, t_dt, q_dt) consistent with
    the FV3 discretization and definition of the prognostic variables
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid_data: GridData,
        namelist,
        comm: pace.util.CubedSphereCommunicator,
        grid_info: DriverGridData,
    ):
        grid_indexing = stencil_factory.grid_indexing
        self.comm = comm
        self._moist_cv = stencil_factory.from_origin_domain(
            moist_cv,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(0, 0, 1)),
        )
        self._update_pressure_and_surface_winds = stencil_factory.from_origin_domain(
            update_pressure_and_surface_winds,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(0, 0, 1)),
        )
        self._AGrid2DGridPhysics = AGrid2DGridPhysics(
            stencil_factory, comm.partitioner, comm.rank, namelist, grid_info
        )
        self._do_cubed_to_latlon = CubedToLatLon(
            stencil_factory,
            grid_data,
            order=namelist.c2l_ord,
        )
        self.origin = grid_indexing.origin_compute()
        self.extent = grid_indexing.domain_compute()
        shape = grid_indexing.max_shape
        full_3Dfield_1pts_halo_spec = grid_indexing.get_quantity_halo_spec(
            shape,
            self.origin,
            dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            n_halo=1,
            backend=stencil_factory.backend,
        )
        self._udt_halo_updater = self.comm.get_scalar_halo_updater(
            [full_3Dfield_1pts_halo_spec]
        )
        self._vdt_halo_updater = self.comm.get_scalar_halo_updater(
            [full_3Dfield_1pts_halo_spec]
        )
        # TODO: check if we actually need surface winds
        self._u_srf = utils.make_storage_from_shape(
            shape[0:2], origin=self.origin, init=True, backend=stencil_factory.backend
        )
        self._v_srf = utils.make_storage_from_shape(
            shape[0:2], origin=self.origin, init=True, backend=stencil_factory.backend
        )

    def __call__(
        self,
        state,
        u_dt: pace.util.Quantity,
        v_dt: pace.util.Quantity,
        t_dt: pace.util.Quantity,
        dt: float,
    ):
        self._moist_cv(
            state.qvapor,
            state.qliquid,
            state.qrain,
            state.qsnow,
            state.qice,
            state.qgraupel,
            state.pt,
            t_dt,
            constants.CP_AIR,
            dt,
        )

        self._udt_halo_updater.start([u_dt])
        self._vdt_halo_updater.start([v_dt])
        self._update_pressure_and_surface_winds(
            state.pe,
            state.delp,
            state.peln,
            state.pk,
            constants.KAPPA,
            state.ua,
            state.va,
            state.ps,
            self._u_srf,
            self._v_srf,
        )
        self._udt_halo_updater.wait()
        self._vdt_halo_updater.wait()
        self._AGrid2DGridPhysics(state.u, state.v, u_dt.storage, v_dt.storage)
        self._do_cubed_to_latlon(
            state.u,
            state.v,
            state.ua,
            state.va,
            self.comm,
        )
