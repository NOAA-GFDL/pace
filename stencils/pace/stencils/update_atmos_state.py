from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

import pace.util
from fv3gfs.physics.physics_state import PhysicsState
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import Float, FloatField
from pace.stencils.fv_update_phys import ApplyPhysics2Dycore
from pace.util.grid import DriverGridData, GridData


def fill_gfs(pe: FloatField, q: FloatField, q_min: Float):

    with computation(BACKWARD), interval(0, -3):
        if q[0, 0, 1] < q_min:
            q = q[0, 0, 0] + (q[0, 0, 1] - q_min) * (pe[0, 0, 2] - pe[0, 0, 1]) / (
                pe[0, 0, 1] - pe[0, 0, 0]
            )

    with computation(BACKWARD), interval(1, -3):
        if q[0, 0, 0] < q_min:
            q = q_min

    with computation(FORWARD), interval(1, -2):
        if q[0, 0, -1] < 0.0:
            q = q[0, 0, 0] + q[0, 0, -1] * (pe[0, 0, 0] - pe[0, 0, -1]) / (
                pe[0, 0, 1] - pe[0, 0, 0]
            )

    with computation(FORWARD), interval(0, -2):
        if q[0, 0, 0] < 0.0:
            q = 0.0


def prepare_tendencies_and_update_tracers(
    u_dt: FloatField,
    v_dt: FloatField,
    pt_dt: FloatField,
    u_t1: FloatField,
    v_t1: FloatField,
    physics_updated_pt: FloatField,
    physics_updated_specific_humidity: FloatField,
    physics_updated_qliquid: FloatField,
    physics_updated_qrain: FloatField,
    physics_updated_qsnow: FloatField,
    physics_updated_qice: FloatField,
    physics_updated_qgraupel: FloatField,
    u_t0: FloatField,
    v_t0: FloatField,
    pt_t0: FloatField,
    qvapor_t0: FloatField,
    qliquid_t0: FloatField,
    qrain_t0: FloatField,
    qsnow_t0: FloatField,
    qice_t0: FloatField,
    qgraupel_t0: FloatField,
    prsi: FloatField,
    delp: FloatField,
    rdt: Float,
):
    """Gather tendencies and adjust dycore tracers values
    GFS total air mass = dry_mass + water_vapor (condensate excluded)
    GFS mixing ratios  = tracer_mass / (dry_mass + vapor_mass)
    FV3 total air mass = dry_mass + [water_vapor + condensate ]
    FV3 mixing ratios  = tracer_mass / (dry_mass+vapor_mass+cond_mass)
    """
    with computation(PARALLEL), interval(0, -1):
        u_dt += (u_t1 - u_t0) * rdt
        v_dt += (v_t1 - v_t0) * rdt
        pt_dt += (physics_updated_pt - pt_t0) * rdt
        dp = prsi[0, 0, 1] - prsi[0, 0, 0]
        qwat_qv = dp * physics_updated_specific_humidity
        qwat_ql = dp * physics_updated_qliquid
        qwat_qr = dp * physics_updated_qrain
        qwat_qs = dp * physics_updated_qsnow
        qwat_qi = dp * physics_updated_qice
        qwat_qg = dp * physics_updated_qgraupel
        qt = qwat_qv + qwat_ql + qwat_qr + qwat_qs + qwat_qi + qwat_qg
        q_sum = qvapor_t0 + qliquid_t0 + qrain_t0 + qsnow_t0 + qice_t0 + qgraupel_t0
        q0 = delp * (1.0 - q_sum) + qt
        delp = q0
        qvapor_t0 = qwat_qv / q0
        qliquid_t0 = qwat_ql / q0
        qrain_t0 = qwat_qr / q0
        qsnow_t0 = qwat_qs / q0
        qice_t0 = qwat_qi / q0
        qgraupel_t0 = qwat_qg / q0


class UpdateAtmosphereState:
    """Fortran name is atmosphere_state_update
    This is an API to apply tendencies and compute a consistent prognostic state.
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid_data: GridData,
        namelist,
        comm: pace.util.CubedSphereCommunicator,
        grid_info: DriverGridData,
        quantity_factory: pace.util.QuantityFactory,
    ):
        grid_indexing = stencil_factory.grid_indexing
        self.namelist = namelist
        origin = grid_indexing.origin_compute()
        shape = grid_indexing.domain_full(add=(1, 1, 1))
        self._rdt = 1.0 / Float(self.namelist.dt_atmos)

        self._prepare_tendencies_and_update_tracers = (
            stencil_factory.from_origin_domain(
                prepare_tendencies_and_update_tracers,
                origin=grid_indexing.origin_compute(),
                domain=grid_indexing.domain_compute(add=(0, 0, 1)),
            )
        )

        dims = [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM]
        self._u_dt = quantity_factory.zeros(dims, "m/s^2", dtype=float)
        self._v_dt = quantity_factory.zeros(dims, "m/s^2", dtype=float)
        self._pt_dt = quantity_factory.zeros(dims, "degK/s", dtype=float)
        self._fill_GFS = stencil_factory.from_origin_domain(
            fill_gfs,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(0, 0, 1)),
        )
        self._apply_physics2dycore = ApplyPhysics2Dycore(
            stencil_factory,
            grid_data,
            self.namelist,
            comm,
            grid_info,
        )

    def __call__(
        self,
        dycore_state,
        phy_state: PhysicsState,
    ):
        self._fill_GFS(
            phy_state.prsi, phy_state.physics_updated_specific_humidity, 1.0e-9
        )
        self._prepare_tendencies_and_update_tracers(
            self._u_dt,
            self._v_dt,
            self._pt_dt,
            phy_state.physics_updated_ua,
            phy_state.physics_updated_va,
            phy_state.physics_updated_pt,
            phy_state.physics_updated_specific_humidity,
            phy_state.physics_updated_qliquid,
            phy_state.physics_updated_qrain,
            phy_state.physics_updated_qsnow,
            phy_state.physics_updated_qice,
            phy_state.physics_updated_qgraupel,
            phy_state.ua,
            phy_state.va,
            phy_state.pt,
            dycore_state.qvapor,
            dycore_state.qliquid,
            dycore_state.qrain,
            dycore_state.qsnow,
            dycore_state.qice,
            dycore_state.qgraupel,
            phy_state.prsi,
            dycore_state.delp,
            self._rdt,
        )
        self._apply_physics2dycore(
            dycore_state,
            self._u_dt,
            self._v_dt,
            self._pt_dt,
        )
