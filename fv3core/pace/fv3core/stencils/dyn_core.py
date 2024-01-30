from typing import Dict, Mapping, Optional

from dace.frontend.python.interface import nounroll as dace_nounroll
from gt4py.cartesian.gtscript import (
    __INLINED,
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import pace.fv3core.stencils.basic_operations as basic
import pace.fv3core.stencils.d_sw as d_sw
import pace.fv3core.stencils.nh_p_grad as nh_p_grad
import pace.fv3core.stencils.pe_halo as pe_halo
import pace.fv3core.stencils.ray_fast as ray_fast
import pace.fv3core.stencils.temperature_adjust as temperature_adjust
import pace.fv3core.stencils.updatedzc as updatedzc
import pace.fv3core.stencils.updatedzd as updatedzd
import pace.util
import pace.util as fv3util
import pace.util.constants as constants
from pace.dsl.dace.orchestration import dace_inhibitor, orchestrate
from pace.dsl.dace.wrapped_halo_exchange import WrappedHaloUpdater
from pace.dsl.stencil import GridIndexing, StencilFactory
from pace.dsl.typing import Float, FloatField, FloatFieldIJ
from pace.fv3core._config import AcousticDynamicsConfig
from pace.fv3core.dycore_state import DycoreState
from pace.fv3core.stencils.c_sw import CGridShallowWaterDynamics
from pace.fv3core.stencils.del2cubed import HyperdiffusionDamping
from pace.fv3core.stencils.pk3_halo import PK3Halo
from pace.fv3core.stencils.riem_solver3 import NonhydrostaticVerticalSolver
from pace.fv3core.stencils.riem_solver_c import NonhydrostaticVerticalSolverCGrid
from pace.util import (
    X_DIM,
    X_INTERFACE_DIM,
    Y_DIM,
    Y_INTERFACE_DIM,
    Z_DIM,
    Z_INTERFACE_DIM,
)
from pace.util.grid import DampingCoefficients, GridData


HUGE_R = 1.0e40


def zero_data(
    mfxd: FloatField,
    mfyd: FloatField,
    cxd: FloatField,
    cyd: FloatField,
    heat_source: FloatField,
    diss_estd: FloatField,
    first_timestep: bool,
):
    """
    Args:
        mfxd (out): mass flux in x direction
        mfyd (out): mass flux in y direction
        cxd (out): courant number in x direction
        cyd (out): courant number in y direction
        heat_source (out): heat accumulated from diffusion of kinetic energy
            gets applied at the end of all acoustic steps
        diss_estd (out): how much energy is dissipated, is mainly captured
            to send to the stochastic physics (in contrast to heat_source)
        first_timestep (in): is this the first acoustic timestep?
    """
    with computation(PARALLEL), interval(...):
        mfxd = 0.0
        mfyd = 0.0
        cxd = 0.0
        cyd = 0.0
        if first_timestep:
            with horizontal(region[3:-3, 3:-3]):
                heat_source = 0.0
                diss_estd = 0.0


def gz_from_surface_height_and_thicknesses(
    zs: FloatFieldIJ, delz: FloatField, gz: FloatField
):
    """
    Args:
        zs (in): surface height
        delz (in): layer thickness
        gz (out): geopotential height
    """
    with computation(BACKWARD):
        with interval(-1, None):
            gz[0, 0, 0] = zs
        with interval(0, -1):
            gz[0, 0, 0] = gz[0, 0, 1] - delz


def interface_pressure_from_toa_pressure_and_thickness(
    delp: FloatField, pem: FloatField, ptop: Float
):
    """
    Args:
        delp (in): pressure thickness of atmospheric layer
        pem (out): interface pressure
        ptop (in): pressure at top of atmosphere
    """
    with computation(FORWARD):
        with interval(0, 1):
            pem[0, 0, 0] = ptop
        with interval(1, None):
            pem[0, 0, 0] = pem[0, 0, -1] + delp


def compute_geopotential(zh: FloatField, gz: FloatField):
    with computation(PARALLEL), interval(...):
        gz = zh * constants.GRAV


def p_grad_c_stencil(
    rdxc: FloatFieldIJ,
    rdyc: FloatFieldIJ,
    uc: FloatField,
    vc: FloatField,
    delpc: FloatField,
    pkc: FloatField,
    gz: FloatField,
    dt2: Float,
):
    """
    Update C-grid winds from the backwards-in-time pressure gradient force

    When this is run the C-grid winds have almost been completely
    updated by computing the momentum equation terms, but the pressure
    gradient force term has not yet been applied. This stencil completes
    the equation and Arakawa C-grid winds have been advected half a timestep
    upon completing this stencil.

    Args:
        rdxc (in):
        rdyc (in):
        uc (inout): x-velocity on the C-grid, has been updated due to advection
            but not yet due to pressure gradient force
        vc (inout): y-velocity on the C-grid, has been updated due to advection
            but not yet due to pressure gradient force
        delpc (in): vertical delta in pressure
        pkc (in): pressure if non-hydrostatic,
            (edge pressure)**(moist kappa) if hydrostatic
        gz (in):  height of the model grid cells (m)
        dt2 (in): half a model timestep (for C-grid update) in seconds
    """
    from __externals__ import hydrostatic

    # derivation in Lin 1997 https://doi.org/10.1002/qj.49712354214
    # FV3 documentation Section 6.6

    with computation(PARALLEL), interval(...):
        if __INLINED(hydrostatic):
            wk = pkc[0, 0, 1] - pkc
        else:
            wk = delpc
        # wk is pressure gradient
        uc = uc + dt2 * rdxc / (wk[-1, 0, 0] + wk) * (
            (gz[-1, 0, 1] - gz) * (pkc[0, 0, 1] - pkc[-1, 0, 0])
            + (gz[-1, 0, 0] - gz[0, 0, 1]) * (pkc[-1, 0, 1] - pkc)
        )

        vc = vc + dt2 * rdyc / (wk[0, -1, 0] + wk) * (
            (gz[0, -1, 1] - gz) * (pkc[0, 0, 1] - pkc[0, -1, 0])
            + (gz[0, -1, 0] - gz[0, 0, 1]) * (pkc[0, -1, 1] - pkc)
        )


def get_nk_heat_dissipation(
    config: d_sw.DGridShallowWaterLagrangianDynamicsConfig, npz: int
) -> int:
    # determines whether to convert dissipated kinetic energy into heat in the full
    # column, not at all, or in 1 or 2 of the top of atmosphere sponge layers
    if config.convert_ke or config.vtdm4 > 1.0e-4:
        nk_heat_dissipation = npz
    else:
        if config.d2_bg_k1 < 1.0e-3:
            nk_heat_dissipation = 0
        else:
            if config.d2_bg_k2 < 1.0e-3:
                nk_heat_dissipation = 1
            else:
                nk_heat_dissipation = 2
    return nk_heat_dissipation


def dyncore_temporaries(
    quantity_factory: pace.util.QuantityFactory,
) -> Mapping[str, pace.util.Quantity]:
    temporaries: Dict[str, pace.util.Quantity] = {}
    for name in ["ut", "vt", "gz", "zh", "pem", "pkc", "pk3", "heat_source", "cappa"]:
        # TODO: the dimensions of ut and vt may not be correct,
        #       because they are not used. double-check and correct as needed.
        temporaries[name] = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )
    for name in ["gz", "pkc", "zh"]:
        temporaries[name] = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM, Z_INTERFACE_DIM],
            units="unknown",
            dtype=Float,
        )
    temporaries["divgd"] = quantity_factory.zeros(
        dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
        units="unknown",
        dtype=Float,
    )
    temporaries["ws3"] = quantity_factory.zeros(
        dims=[X_DIM, Y_DIM],
        units="unknown",
        dtype=Float,
    )
    for name in ["crx", "xfx"]:
        temporaries[name] = quantity_factory.zeros(
            dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )
    for name in ["cry", "yfx"]:
        temporaries[name] = quantity_factory.zeros(
            dims=[X_DIM, Y_INTERFACE_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )
    return temporaries


class AcousticDynamics:
    """
    Fortran name is dyn_core
    Peforms the Lagrangian acoustic dynamics described by Lin 2004
    """

    class _HaloUpdaters(object):
        """Encapsulate all HaloUpdater objects"""

        def __init__(
            self,
            comm: pace.util.Communicator,
            grid_indexing: GridIndexing,
            quantity_factory: pace.util.QuantityFactory,
            state: DycoreState,
            cappa: pace.util.Quantity,
            gz: pace.util.Quantity,
            zh: pace.util.Quantity,
            divgd: pace.util.Quantity,
            heat_source: pace.util.Quantity,
            pkc: pace.util.Quantity,
        ):
            # Define the memory specification required
            # Those can be re-used as they are read-only descriptors
            full_size_xyz_halo_spec = quantity_factory.get_quantity_halo_spec(
                dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
                n_halo=grid_indexing.n_halo,
                dtype=Float,
            )
            full_size_xyiz_halo_spec = quantity_factory.get_quantity_halo_spec(
                dims=[fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
                n_halo=grid_indexing.n_halo,
                dtype=Float,
            )
            full_size_xiyz_halo_spec = quantity_factory.get_quantity_halo_spec(
                dims=[fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
                n_halo=grid_indexing.n_halo,
                dtype=Float,
            )
            full_size_xyzi_halo_spec = quantity_factory.get_quantity_halo_spec(
                dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM],
                n_halo=grid_indexing.n_halo,
                dtype=Float,
            )
            full_size_xiyiz_halo_spec = quantity_factory.get_quantity_halo_spec(
                dims=[fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
                n_halo=grid_indexing.n_halo,
                dtype=Float,
            )

            # Build the HaloUpdater. We could build one updater per specification group
            # but because of call overlap between different variable, we kept the
            # straighforward solution of one HaloUpdater per group of updated variable.
            # It also makes the code in call() more readable
            # [DaCe] Wrapping call to a DaCe readable halo updater
            #        Biggest parsing issue is that DaCe cannot do
            #        quantities at runtime paradigm
            self.q_con__cappa = WrappedHaloUpdater(
                comm.get_scalar_halo_updater([full_size_xyz_halo_spec] * 2),
                dict(q_con=state.q_con, cappa=cappa),
                ["q_con", "cappa"],
            )
            self.delp__pt = WrappedHaloUpdater(
                comm.get_scalar_halo_updater([full_size_xyz_halo_spec] * 2),
                state,
                ["delp", "pt"],
            )
            self.u__v = WrappedHaloUpdater(
                comm.get_vector_halo_updater(
                    [full_size_xyiz_halo_spec], [full_size_xiyz_halo_spec]
                ),
                state,
                ["u"],
                ["v"],
            )
            self.w = WrappedHaloUpdater(
                comm.get_scalar_halo_updater([full_size_xyz_halo_spec]),
                state,
                ["w"],
            )
            self.gz = WrappedHaloUpdater(
                comm.get_scalar_halo_updater([full_size_xyzi_halo_spec]),
                {"gz": gz},
                ["gz"],
            )
            self.delp__pt__q_con = WrappedHaloUpdater(
                comm.get_scalar_halo_updater([full_size_xyz_halo_spec] * 3),
                state,
                ["delp", "pt", "q_con"],
            )
            self.zh = WrappedHaloUpdater(
                comm.get_scalar_halo_updater([full_size_xyzi_halo_spec]),
                {"zh": zh},
                ["zh"],
            )
            self.divgd = WrappedHaloUpdater(
                comm.get_scalar_halo_updater([full_size_xiyiz_halo_spec]),
                {"divgd": divgd},
                ["divgd"],
            )
            self.heat_source = WrappedHaloUpdater(
                comm.get_scalar_halo_updater([full_size_xyz_halo_spec]),
                {"heat_source": heat_source},
                ["heat_source"],
            )
            if grid_indexing.domain[0] == grid_indexing.domain[1]:
                full_3Dfield_2pts_halo_spec = quantity_factory.get_quantity_halo_spec(
                    dims=[fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM],
                    n_halo=2,
                    dtype=Float,
                )
                self.pkc = WrappedHaloUpdater(
                    comm.get_scalar_halo_updater([full_3Dfield_2pts_halo_spec]),
                    {"pkc": pkc},
                    ["pkc"],
                )
            else:
                self.pkc = comm.get_scalar_halo_updater([full_size_xyzi_halo_spec])
            self.uc__vc = WrappedHaloUpdater(
                comm.get_vector_halo_updater(
                    [full_size_xiyz_halo_spec], [full_size_xyiz_halo_spec]
                ),
                state,
                ["uc"],
                ["vc"],
            )
            self.interface_uc__vc = WrappedHaloUpdater(
                None, state, ["u"], ["v"], comm=comm
            )

    def __init__(
        self,
        comm: pace.util.Communicator,
        stencil_factory: StencilFactory,
        quantity_factory: pace.util.QuantityFactory,
        grid_data: GridData,
        damping_coefficients: DampingCoefficients,
        grid_type,
        nested,
        stretched_grid,
        config: AcousticDynamicsConfig,
        phis: FloatFieldIJ,
        wsd: FloatFieldIJ,
        state,  # [DaCe] hack to get around quantity as parameters for halo updates
        checkpointer: Optional[pace.util.Checkpointer] = None,
    ):
        """
        Args:
            comm: object for tile or cubed-sphere inter-process communication
            stencil_factory: creates stencils
            quantity_factory: creates quantities
            grid_data: metric terms defining the grid
            damping_coefficients: damping configuration
            grid_type: grid geometry used
            nested: if the grid contains a nested, high-res region
            stretched_grid: if the grid is stretched so tile faces cover different areas
            config: configuration settings
            pfull: atmospheric Eulerian grid reference pressure (Pa)
            phis: surface geopotential height
            checkpointer: if given, used to perform operations on model data
                at specific points in model execution, such as testing against
                reference data
        """
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_compiletime_args=["state"],
        )

        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="_checkpoint_csw",
            dace_compiletime_args=["state", "tag"],
        )

        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="_checkpoint_dsw_in",
            dace_compiletime_args=["state", "tag"],
        )

        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="_checkpoint_dsw_out",
            dace_compiletime_args=["state", "tag"],
        )

        self.call_checkpointer = checkpointer is not None
        if checkpointer is None:
            self.checkpointer: pace.util.Checkpointer = pace.util.NullCheckpointer()
        else:
            self.checkpointer = checkpointer
        grid_indexing = stencil_factory.grid_indexing
        self.config = config
        if config.d_ext != 0:
            raise RuntimeError("Acoustics (dyn_core): d_ext != 0 is not implemented")
        if config.beta != 0:
            raise RuntimeError("Acoustics (dyn_core): beta != 0 is not implemented")
        if config.use_logp:
            raise RuntimeError("Acoustics (dyn_core): use_logp=True is not implemented")
        self._da_min = damping_coefficients.da_min
        self.grid_data = grid_data
        self._ptop = grid_data.ptop
        self._pfull = grid_data.p
        self._wsd = wsd
        self._nk_heat_dissipation = get_nk_heat_dissipation(
            config.d_grid_shallow_water,
            npz=grid_indexing.domain[2],
        )
        self.nonhydrostatic_pressure_gradient = (
            nh_p_grad.NonHydrostaticPressureGradient(
                stencil_factory,
                quantity_factory=quantity_factory,
                grid_data=grid_data,
                grid_type=config.grid_type,
            )
        )
        self._akap = Float(constants.KAPPA)

        temporaries = dyncore_temporaries(quantity_factory)
        self._heat_source = temporaries["heat_source"]
        self._divgd = temporaries["divgd"]
        self._gz = temporaries["gz"]
        self._pkc = temporaries["pkc"]
        self._zh = temporaries["zh"]
        self.cappa = temporaries["cappa"]
        self._ut = temporaries["ut"]
        self._vt = temporaries["vt"]
        self._pem = temporaries["pem"]
        self._pk3 = temporaries["pk3"]
        self._crx = temporaries["crx"]
        self._cry = temporaries["cry"]
        self._xfx = temporaries["xfx"]
        self._yfx = temporaries["yfx"]
        self._ws3 = temporaries["ws3"]

        if not config.hydrostatic:
            self._pk3.data[:] = HUGE_R

        column_namelist = d_sw.get_column_namelist(
            config.d_grid_shallow_water, quantity_factory=quantity_factory
        )
        if not config.hydrostatic:
            # To write lower dimensional storages, these need to be 3D
            # then converted to lower dimensional
            self._dp_ref = grid_data.dp_ref
            self._zs = quantity_factory.zeros(
                [X_DIM, Y_DIM],
                units="m",
                dtype=Float,
            )
            self._zs.data[:] = self._zs.np.asarray(
                phis.data / constants.GRAV, dtype=self._zs.data.dtype
            )

            self.update_height_on_d_grid = updatedzd.UpdateHeightOnDGrid(
                stencil_factory,
                quantity_factory=quantity_factory,
                damping_coefficients=damping_coefficients,
                grid_data=grid_data,
                grid_type=grid_type,
                hord_tm=config.hord_tm,
                column_namelist=column_namelist,
            )
            self.vertical_solver = NonhydrostaticVerticalSolver(
                stencil_factory,
                quantity_factory=quantity_factory,
                config=config.riemann,
            )
            self.vertical_solver_cgrid = NonhydrostaticVerticalSolverCGrid(
                stencil_factory, quantity_factory=quantity_factory, p_fac=config.p_fac
            )
            origin, domain = grid_indexing.get_origin_domain(
                [X_DIM, Y_DIM, Z_INTERFACE_DIM], halos=(2, 2)
            )
            self._compute_geopotential_stencil = stencil_factory.from_origin_domain(
                compute_geopotential,
                origin=origin,
                domain=domain,
            )
        self.dgrid_shallow_water_lagrangian_dynamics = (
            d_sw.DGridShallowWaterLagrangianDynamics(
                stencil_factory,
                quantity_factory=quantity_factory,
                grid_data=grid_data,
                damping_coefficients=damping_coefficients,
                column_namelist=column_namelist,
                nested=nested,
                stretched_grid=stretched_grid,
                config=config.d_grid_shallow_water,
            )
        )

        self.cgrid_shallow_water_lagrangian_dynamics = CGridShallowWaterDynamics(
            stencil_factory,
            quantity_factory=quantity_factory,
            grid_data=grid_data,
            nested=nested,
            grid_type=config.grid_type,
            nord=config.nord,
        )

        self._gz_from_surface_height_and_thickness = stencil_factory.from_origin_domain(
            gz_from_surface_height_and_thicknesses,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(0, 0, 1)),
        )
        self._interface_pressure_from_toa_pressure_and_thickness = (
            stencil_factory.from_origin_domain(
                interface_pressure_from_toa_pressure_and_thickness,
                origin=grid_indexing.origin_compute(add=(-1, -1, 0)),
                domain=grid_indexing.domain_compute(add=(2, 2, 0)),
            )
        )

        self._p_grad_c = stencil_factory.from_origin_domain(
            p_grad_c_stencil,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(1, 1, 0)),
            externals={"hydrostatic": config.hydrostatic},
        )

        self.update_geopotential_height_on_c_grid = (
            updatedzc.UpdateGeopotentialHeightOnCGrid(
                stencil_factory,
                quantity_factory=quantity_factory,
                area=grid_data.area,
                dp_ref=grid_data.dp_ref,
                grid_type=config.grid_type,
            )
        )

        self._zero_data = stencil_factory.from_origin_domain(
            zero_data,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(),
        )
        ax_offsets_pe = grid_indexing.axis_offsets(
            grid_indexing.origin_full(),
            grid_indexing.domain_full(add=(0, 0, 1)),
        )
        self._edge_pe_stencil = stencil_factory.from_origin_domain(
            pe_halo.edge_pe,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(0, 0, 1)),
            externals={**ax_offsets_pe},
            skip_passes=("PruneKCacheFills",),
        )
        """The stencil object responsible for updating the interface pressure"""

        self._do_del2cubed = self._nk_heat_dissipation != 0 and config.d_con > 1.0e-5

        if self._do_del2cubed:
            nf_ke = min(3, config.nord + 1)
            self._hyperdiffusion = HyperdiffusionDamping(
                stencil_factory,
                quantity_factory=quantity_factory,
                damping_coefficients=damping_coefficients,
                rarea=grid_data.rarea,
                nmax=nf_ke,
            )
        if config.rf_fast:
            self._rayleigh_damping = ray_fast.RayleighDamping(
                stencil_factory,
                rf_cutoff=config.rf_cutoff,
                tau=config.tau,
                hydrostatic=config.hydrostatic,
            )
        self._apply_diffusive_heating = stencil_factory.from_origin_domain(
            temperature_adjust.apply_diffusive_heating,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.restrict_vertical(
                nk=self._nk_heat_dissipation
            ).domain_compute(),
        )
        self._pk3_halo = PK3Halo(stencil_factory, quantity_factory)
        self._copy_stencil = stencil_factory.from_origin_domain(
            basic.copy_defn,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(0, 0, 1)),
        )

        # Halo updaters
        self._halo_updaters = AcousticDynamics._HaloUpdaters(
            comm,
            grid_indexing,
            quantity_factory,
            state,
            cappa=self.cappa,
            gz=self._gz,
            zh=self._zh,
            divgd=self._divgd,
            heat_source=self._heat_source,
            pkc=self._pkc,
        )

    # See divergence_damping.py, _get_da_min for explanation of this function
    @dace_inhibitor
    def _get_da_min(self) -> float:
        return self._da_min

    def _checkpoint_csw(self, state: DycoreState, tag: str):
        if self.call_checkpointer:
            self.checkpointer(
                f"C_SW-{tag}",
                delpd=state.delp,
                ptd=state.pt,
                ud=state.u,
                vd=state.v,
                wd=state.w,
                ucd=state.uc,
                vcd=state.vc,
                uad=state.ua,
                vad=state.va,
                utd=self._ut,
                vtd=self._vt,
                divgdd=self._divgd,
            )

    def _checkpoint_dsw_in(self, state: DycoreState):
        if self.call_checkpointer:
            self.checkpointer(
                "D_SW-In",
                ucd=state.uc,
                vcd=state.vc,
                wd=state.w,
                # delpc is a temporary and not a variable in D_SW savepoint
                delpcd=self._vt,
                delpd=state.delp,
                ud=state.u,
                vd=state.v,
                ptd=state.pt,
                uad=state.ua,
                vad=state.va,
                zhd=self._zh,
                divgdd=self._divgd,
                xfxd=self._xfx,
                yfxd=self._yfx,
                mfxd=state.mfxd,
                mfyd=state.mfyd,
            )

    def _checkpoint_dsw_out(self, state: DycoreState):
        if self.call_checkpointer:
            self.checkpointer(
                "D_SW-Out",
                ucd=state.uc,
                vcd=state.vc,
                wd=state.w,
                delpcd=self._vt,
                delpd=state.delp,
                ud=state.u,
                vd=state.v,
                ptd=state.pt,
                uad=state.ua,
                vad=state.va,
                divgdd=self._divgd,
                xfxd=self._xfx,
                yfxd=self._yfx,
                mfxd=state.mfxd,
                mfyd=state.mfyd,
            )

    # TODO: fix me - we shouldn't need a function here, Dace is fudging the types
    # See https://github.com/GEOS-ESM/pace/issues/9
    @dace_inhibitor
    def dt_acoustic_substep(self, timestep: Float) -> Float:
        return timestep / self.config.n_split

    # TODO: Same as above
    @dace_inhibitor
    def dt2(self, dt_acoustic_substep: Float) -> Float:
        return 0.5 * dt_acoustic_substep

    def __call__(
        self,
        state: DycoreState,
        timestep: Float,  # time to step forward by in seconds
        n_map=1,  # [DaCe] replaces state.n_map
    ):
        # u, v, w, delz, delp, pt, pe, pk, phis, wsd, omga, ua, va, uc, vc, mfxd,
        # mfyd, cxd, cyd, pkz, peln, q_con, ak, bk, diss_estd, cappa, mdt, n_split,
        # akap, ptop, n_map, comm):
        end_step = n_map == self.config.k_split
        # dt = state.mdt / self.config.n_split
        dt_acoustic_substep: Float = self.dt_acoustic_substep(timestep)
        dt2: Float = self.dt2(dt_acoustic_substep)
        n_split = self.config.n_split
        # NOTE: In Fortran model the halo update starts happens in fv_dynamics, not here
        self._halo_updaters.q_con__cappa.start()
        self._halo_updaters.delp__pt.start()
        self._halo_updaters.u__v.start()
        self._halo_updaters.q_con__cappa.wait()

        self._zero_data(
            state.mfxd,
            state.mfyd,
            state.cxd,
            state.cyd,
            self._heat_source,
            state.diss_estd,
            n_map == 1,
        )

        # "acoustic" loop
        # called this because its timestep is usually limited by horizontal sound-wave
        # processes. Note this is often not the limiting factor near the poles, where
        # the speed of the polar night jets can exceed two-thirds of the speed of sound.
        for it in dace_nounroll(range(n_split)):
            # the Lagrangian dynamics have two parts. First we advance the C-grid winds
            # by half a time step (c_sw). Then the C-grid winds are used to define
            # advective fluxes to advance the D-grid prognostic fields a full time step
            # (the rest of the routines).
            #
            # Along-surface flux terms (mass, heat, vertical momentum, vorticity,
            # kinetic energy gradient terms) are evaluated forward-in-time.
            #
            # The pressure gradient force and elastic terms are then evaluated
            # backwards-in-time, to improve stability.
            remap_step = False
            if self.config.breed_vortex_inline or (it == n_split - 1):
                remap_step = True
            if not self.config.hydrostatic:
                self._halo_updaters.w.start()
                if it == 0:
                    self._gz_from_surface_height_and_thickness(
                        self._zs,
                        state.delz,
                        self._gz,
                    )
                    self._halo_updaters.gz.start()
            if it == 0:
                self._halo_updaters.delp__pt.wait()

            if it == n_split - 1 and end_step:
                if self.config.use_old_omega:
                    self._interface_pressure_from_toa_pressure_and_thickness(
                        state.delp,
                        self._pem,
                        self._ptop,
                    )

            self._halo_updaters.u__v.wait()
            if not self.config.hydrostatic:
                self._halo_updaters.w.wait()

            # compute the c-grid winds at t + 1/2 timestep
            self._checkpoint_csw(state, tag="In")
            self.cgrid_shallow_water_lagrangian_dynamics(
                state.delp,
                state.pt,
                state.u,
                state.v,
                state.w,
                state.uc,
                state.vc,
                state.ua,
                state.va,
                self._ut,
                self._vt,
                self._divgd,
                state.omga,
                dt2,
            )
            self._checkpoint_csw(state, tag="Out")

            # TODO: Computing the pressure gradient outside of C_SW was originally done
            # so that we could transpose into a vertical-first memory ordering for the
            # gz computation, now that we have gt4py we should pull this into C_SW.
            if self.config.nord > 0:
                self._halo_updaters.divgd.start()
            if not self.config.hydrostatic:
                # TODO: is there some way we can avoid aliasing gz and zh, so that
                # gz is always a geopotential and zh is always a height?
                if it == 0:
                    self._halo_updaters.gz.wait()
                    self._copy_stencil(
                        self._gz,
                        self._zh,
                    )
                else:
                    self._copy_stencil(
                        self._zh,
                        self._gz,
                    )
            if not self.config.hydrostatic:
                self.update_geopotential_height_on_c_grid(
                    self._zs, self._ut, self._vt, self._gz, self._ws3, dt2
                )
                # TODO (floriand): Due to DaCe VRAM pooling creating a memory
                # leak with the usage pattern of those two fields
                # We use the C_SW internal to workaround it e.g.:
                #  - self.cgrid_shallow_water_lagrangian_dynamics.delpc
                #  - self.cgrid_shallow_water_lagrangian_dynamics.ptc
                # DaCe has already a fix on their side and it awaits release
                # issue
                self.vertical_solver_cgrid(
                    dt2,
                    self.cappa,
                    self._ptop,
                    state.phis,
                    self._ws3,
                    self.cgrid_shallow_water_lagrangian_dynamics.ptc,
                    state.q_con,
                    self.cgrid_shallow_water_lagrangian_dynamics.delpc,
                    self._gz,
                    self._pkc,
                    state.omga,
                )

            self._p_grad_c(
                self.grid_data.rdxc,
                self.grid_data.rdyc,
                state.uc,
                state.vc,
                self.cgrid_shallow_water_lagrangian_dynamics.delpc,
                self._pkc,
                self._gz,
                dt2,
            )
            self._halo_updaters.uc__vc.start()
            if self.config.nord > 0:
                self._halo_updaters.divgd.wait()
            self._halo_updaters.uc__vc.wait()
            # use the computed c-grid winds to evolve the d-grid winds forward
            # by 1 timestep
            self._checkpoint_dsw_in(state)
            self.dgrid_shallow_water_lagrangian_dynamics(
                self._vt,
                state.delp,
                state.pt,
                state.u,
                state.v,
                state.w,
                state.uc,
                state.vc,
                state.ua,
                state.va,
                self._divgd,
                state.mfxd,
                state.mfyd,
                state.cxd,
                state.cyd,
                self._crx,
                self._cry,
                self._xfx,
                self._yfx,
                state.q_con,
                self._zh,
                self._heat_source,
                state.diss_estd,
                dt_acoustic_substep,
            )
            self._checkpoint_dsw_out(state)
            # note that uc and vc are not needed at all past this point.
            # they will be re-computed from scratch on the next acoustic timestep.

            self._halo_updaters.delp__pt__q_con.update()

            # Not used unless we implement other betas and alternatives to nh_p_grad
            # if self.namelist.d_ext > 0:
            #    raise 'Unimplemented namelist option d_ext > 0'

            # TODO: should the dycore have hydrostatic and non-hydrostatic modes,
            # or would we make a new class for the non-hydrostatic mode?
            if not self.config.hydrostatic:
                # without explicit arg names, numpy does not run
                self.update_height_on_d_grid(
                    surface_height=self._zs,
                    height=self._zh,
                    courant_number_x=self._crx,
                    courant_number_y=self._cry,
                    x_area_flux=self._xfx,
                    y_area_flux=self._yfx,
                    ws=self._wsd,
                    dt=dt_acoustic_substep,
                )
                self.vertical_solver(
                    remap_step,
                    dt_acoustic_substep,
                    self.cappa,
                    self._ptop,
                    self._zs,
                    self._wsd,
                    state.delz,
                    state.q_con,
                    state.delp,
                    state.pt,
                    self._zh,
                    state.pe,
                    self._pkc,
                    self._pk3,
                    state.pk,
                    state.peln,
                    state.w,
                )

                self._halo_updaters.zh.start()
                self._halo_updaters.pkc.start()
                if remap_step:
                    # TODO: can this be moved to the start of the remapping routine?
                    self._edge_pe_stencil(state.pe, state.delp, self._ptop)
                if self.config.use_logp:
                    raise NotImplementedError(
                        "unimplemented namelist option use_logp=True"
                    )
                else:
                    self._pk3_halo(self._pk3, state.delp, self._ptop, self._akap)
            if not self.config.hydrostatic:
                self._halo_updaters.zh.wait()
                self._compute_geopotential_stencil(
                    self._zh,
                    self._gz,
                )
                self._halo_updaters.pkc.wait()

                self.nonhydrostatic_pressure_gradient(
                    state.u,
                    state.v,
                    self._pkc,
                    self._gz,
                    self._pk3,
                    state.delp,
                    dt_acoustic_substep,
                    self._ptop,
                    self._akap,
                )

            if self.config.rf_fast:
                # TODO: Pass through ks, or remove, inconsistent representation vs
                # Fortran.
                self._rayleigh_damping(
                    u=state.u,
                    v=state.v,
                    w=state.w,
                    dp=self._dp_ref,
                    pfull=self._pfull,
                    dt=dt_acoustic_substep,
                    ptop=self._ptop,
                )

            if it != n_split - 1:
                # [DaCe] this should be a reuse of
                #        self._halo_updaters.u__v but it creates
                #        parameter generation issues, and therefore has been duplicated
                self._halo_updaters.u__v.start()
            else:
                if self.config.grid_type < 4:
                    self._halo_updaters.interface_uc__vc.interface()

        # we are here

        if self._do_del2cubed:
            self._halo_updaters.heat_source.update()
            # TODO: move dependence on da_min into init of hyperdiffusion class
            da_min: Float = self._get_da_min()
            cd = constants.CNST_0P20 * da_min
            # we want to diffuse the heat source from damping before we apply it,
            # so that we don't reinforce the same grid-scale patterns we're trying
            # to damp
            self._hyperdiffusion(self._heat_source, cd)
            if not self.config.hydrostatic:
                delt_time_factor = abs(dt_acoustic_substep * self.config.delt_max)
                # TODO: it looks like state.pkz is being used as a temporary here,
                # and overwritten at the start of remapping. See if we can make it
                # an internal temporary of this stencil.
                self._apply_diffusive_heating(
                    state.delp,
                    state.delz,
                    self.cappa,
                    self._heat_source,
                    state.pt,
                    delt_time_factor,
                )
