import gt4py.cartesian.gtscript as gtscript
import ndsl.constants as constants
from gt4py.cartesian.gtscript import FORWARD, PARALLEL, computation, exp, interval, log
from ndsl.comm.communicator import Communicator
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.dace.orchestration import orchestrate
from ndsl.dsl.dace.wrapped_halo_exchange import WrappedHaloUpdater
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from ndsl.grid import DriverGridData, GridData
from ndsl.initialization.allocator import QuantityFactory
from ndsl.quantity import Quantity
from ndsl.stencils.c2l_ord import CubedToLatLon

from pace import fv3core
from pace.physics.update.update_dwind_phys import AGrid2DGridPhysics


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
    with computation(PARALLEL), interval(...):
        t_dt = 0.0


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


class ApplyPhysicsToDycore:
    """
    Fortran name is fv_update_phys
    Apply the physics tendencies (u_dt, v_dt, t_dt, q_dt) consistent with
    the FV3 discretization and definition of the prognostic variables
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        grid_data: GridData,
        namelist,
        comm: Communicator,
        grid_info: DriverGridData,
        state: fv3core.DycoreState,
        u_dt: Quantity,
        v_dt: Quantity,
    ):
        self._grid_type = grid_info.grid_type
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_compiletime_args=["state"],
        )
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
            stencil_factory,
            quantity_factory,
            comm.partitioner.tile,
            comm.rank,
            namelist,
            grid_info,
        )
        self._do_cubed_to_latlon = CubedToLatLon(
            state,
            stencil_factory,
            quantity_factory=quantity_factory,
            grid_data=grid_data,
            order=namelist.c2l_ord,
            comm=comm,
            grid_type=self._grid_type,
        )
        origin = grid_indexing.origin_compute()
        shape = grid_indexing.max_shape
        full_3Dfield_1pts_halo_spec = quantity_factory.get_quantity_halo_spec(
            dims=[X_DIM, Y_DIM, Z_DIM],
            n_halo=1,
        )
        self._udt_halo_updater = WrappedHaloUpdater(
            self.comm.get_scalar_halo_updater([full_3Dfield_1pts_halo_spec]),
            {"u_dt": u_dt},
            ["u_dt"],
        )
        self._vdt_halo_updater = WrappedHaloUpdater(
            self.comm.get_scalar_halo_updater([full_3Dfield_1pts_halo_spec]),
            {"v_dt": v_dt},
            ["v_dt"],
        )
        # TODO: check if we actually need surface winds
        self._u_srf = quantity_factory.zeros(dims=[X_DIM, Y_DIM], units="m/s")
        self._v_srf = quantity_factory.zeros(dims=[X_DIM, Y_DIM], units="m/s")

    def __call__(
        self,
        state,
        u_dt,
        v_dt,
        t_dt,
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

        self._udt_halo_updater.start()
        self._vdt_halo_updater.start()
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
        self._AGrid2DGridPhysics(state.u, state.v, u_dt, v_dt)
        self._do_cubed_to_latlon(
            state.u,
            state.v,
            state.ua,
            state.va,
        )
