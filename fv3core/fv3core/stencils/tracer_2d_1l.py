import math

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core._config as spec
import fv3gfs.util
import fv3gfs.util.pace.gt4py_utils as utils
import fv3gfs.util.stencils.fxadv
from fv3gfs.util.pace.typing import FloatField, FloatFieldIJ
from fv3gfs.util.stencil import StencilFactory
from fv3gfs.util.stencils.fvtp2d import (
    FiniteVolumeTransport,
    PreAllocatedCopiedCornersFactory,
)


@gtscript.function
def flux_x(cx, dxa, dy, sin_sg3, sin_sg1, xfx):
    from __externals__ import local_ie, local_is, local_je, local_js

    with horizontal(region[local_is : local_ie + 2, local_js - 3 : local_je + 4]):
        xfx = (
            cx * dxa[-1, 0] * dy * sin_sg3[-1, 0] if cx > 0 else cx * dxa * dy * sin_sg1
        )
    return xfx


@gtscript.function
def flux_y(cy, dya, dx, sin_sg4, sin_sg2, yfx):
    from __externals__ import local_ie, local_is, local_je, local_js

    with horizontal(region[local_is - 3 : local_ie + 4, local_js : local_je + 2]):
        yfx = (
            cy * dya[0, -1] * dx * sin_sg4[0, -1] if cy > 0 else cy * dya * dx * sin_sg2
        )
    return yfx


def flux_compute(
    cx: FloatField,
    cy: FloatField,
    dxa: FloatFieldIJ,
    dya: FloatFieldIJ,
    dx: FloatFieldIJ,
    dy: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    xfx: FloatField,
    yfx: FloatField,
):
    with computation(PARALLEL), interval(...):
        xfx = flux_x(cx, dxa, dy, sin_sg3, sin_sg1, xfx)
        yfx = flux_y(cy, dya, dx, sin_sg4, sin_sg2, yfx)


def cmax_multiply_by_frac(
    cxd: FloatField,
    xfx: FloatField,
    mfxd: FloatField,
    cyd: FloatField,
    yfx: FloatField,
    mfyd: FloatField,
    n_split: int,
):
    """multiply all other inputs in-place by frac."""
    with computation(PARALLEL), interval(...):
        frac = 1.0 / n_split
        cxd = cxd * frac
        xfx = xfx * frac
        mfxd = mfxd * frac
        cyd = cyd * frac
        yfx = yfx * frac
        mfyd = mfyd * frac


def cmax_stencil1(cx: FloatField, cy: FloatField, cmax: FloatField):
    with computation(PARALLEL), interval(...):
        cmax = max(abs(cx), abs(cy))


def cmax_stencil2(
    cx: FloatField, cy: FloatField, sin_sg5: FloatField, cmax: FloatField
):
    with computation(PARALLEL), interval(...):
        cmax = max(abs(cx), abs(cy)) + 1.0 - sin_sg5


def dp_fluxadjustment(
    dp1: FloatField,
    mfx: FloatField,
    mfy: FloatField,
    rarea: FloatFieldIJ,
    dp2: FloatField,
):
    with computation(PARALLEL), interval(...):
        dp2 = dp1 + (mfx - mfx[1, 0, 0] + mfy - mfy[0, 1, 0]) * rarea


@gtscript.function
def adjustment(q, dp1, fx, fy, rarea, dp2):
    return (q * dp1 + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / dp2


def q_adjust(
    q: FloatField,
    dp1: FloatField,
    fx: FloatField,
    fy: FloatField,
    rarea: FloatFieldIJ,
    dp2: FloatField,
):
    with computation(PARALLEL), interval(...):
        q = adjustment(q, dp1, fx, fy, rarea, dp2)


class TracerAdvection:
    """
    Performs horizontal advection on tracers.

    Corresponds to tracer_2D_1L in the Fortran code.
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        transport: FiniteVolumeTransport,
        grid_data,
        comm: fv3gfs.util.CubedSphereCommunicator,
        tracer_count,
    ):
        grid_indexing = stencil_factory.grid_indexing
        self._tracer_count = tracer_count
        self.comm = comm
        self.grid = spec.grid
        self.grid_data = grid_data
        shape = grid_indexing.domain_full(add=(1, 1, 1))
        origin = grid_indexing.origin_compute()
        self._tmp_xfx = utils.make_storage_from_shape(shape, origin)
        self._tmp_yfx = utils.make_storage_from_shape(shape, origin)
        self._tmp_fx = utils.make_storage_from_shape(shape, origin)
        self._tmp_fy = utils.make_storage_from_shape(shape, origin)
        self._tmp_dp = utils.make_storage_from_shape(shape, origin)
        self._tmp_qn2 = self.grid.quantity_wrap(
            utils.make_storage_from_shape(shape, origin),
            units="kg/m^2",
        )

        ax_offsets = fv3gfs.util.axis_offsets(
            self.grid, grid_indexing.origin_full(), grid_indexing.domain_full()
        )
        local_axis_offsets = {}
        for axis_offset_name, axis_offset_value in ax_offsets.items():
            if "local" in axis_offset_name:
                local_axis_offsets[axis_offset_name] = axis_offset_value

        self._flux_compute = stencil_factory.from_origin_domain(
            flux_compute,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(1, 1, 0)),
            externals=local_axis_offsets,
        )
        self._cmax_multiply_by_frac = stencil_factory.from_origin_domain(
            cmax_multiply_by_frac,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(1, 1, 0)),
            externals=local_axis_offsets,
        )
        self._dp_fluxadjustment = stencil_factory.from_origin_domain(
            dp_fluxadjustment,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
            externals=local_axis_offsets,
        )
        self._q_adjust = stencil_factory.from_origin_domain(
            q_adjust,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
            externals=local_axis_offsets,
        )
        self.finite_volume_transport: FiniteVolumeTransport = transport
        # If use AllReduce, will need something like this:
        # self._tmp_cmax = utils.make_storage_from_shape(shape, origin)
        # self._cmax_1 = stencil_factory.from_origin_domain(cmax_stencil1)
        # self._cmax_2 = stencil_factory.from_origin_domain(cmax_stencil2)

        # Setup halo updater for tracers
        tracer_halo_spec = self.grid.get_halo_update_spec(shape, origin, utils.halo)
        self._tracers_halo_updater = self.comm.get_scalar_halo_updater(
            [tracer_halo_spec] * tracer_count
        )
        self._copy_corners = PreAllocatedCopiedCornersFactory(
            stencil_factory=stencil_factory,
            dims=[fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM],
            y_temporary=None,
        )

    def __call__(self, tracers, dp1, mfxd, mfyd, cxd, cyd, mdt):
        if len(tracers) != self._tracer_count:
            raise ValueError(
                f"incorrect number of tracers, {self._tracer_count} was "
                f"specified on init but {len(tracers)} were passed"
            )
        # start HALO update on q (in dyn_core in fortran -- just has started when
        # this function is called...)
        self._flux_compute(
            cxd,
            cyd,
            self.grid_data.dxa,
            self.grid_data.dya,
            self.grid_data.dx,
            self.grid_data.dy,
            self.grid_data.sin_sg1,
            self.grid_data.sin_sg2,
            self.grid_data.sin_sg3,
            self.grid_data.sin_sg4,
            self._tmp_xfx,
            self._tmp_yfx,
        )

        # # TODO for if we end up using the Allreduce and compute cmax globally
        # (or locally). For now, hardcoded.
        # split = int(grid_indexing.domain[2] / 6)
        # self._cmax_1(
        #     cxd, cyd, self._tmp_cmax, origin=grid_indexing.origin_compute(),
        #     domain=(grid_indexing.domain[0], self.grid_indexing.domain[1], split)
        # )
        # self._cmax_2(
        #     cxd,
        #     cyd,
        #     self.grid.sin_sg5,
        #     self._tmp_cmax,
        #     origin=(grid_indexing.isc, self.grid_indexing.jsc, split),
        #     domain=(
        #         grid_indexing.domain[0],
        #         self.grid_indexing.domain[1],
        #         grid_indexing.domain[2] - split + 1
        #     ),
        # )
        # cmax_flat = np.amax(self._tmp_cmax, axis=(0, 1))
        # # cmax_flat is a gt4py storage still, but of dimension [npz+1]...

        # cmax_max_all_ranks = cmax_flat.data
        # # TODO mpi allreduce...
        # # comm.Allreduce(cmax_flat, cmax_max_all_ranks, op=MPI.MAX)

        cmax_max_all_ranks = 2.0
        n_split = math.floor(1.0 + cmax_max_all_ranks)
        # NOTE: cmax is not usually a single value, it varies with k, if return to
        # that, make n_split a column as well

        if n_split > 1.0:
            self._cmax_multiply_by_frac(
                cxd,
                self._tmp_xfx,
                mfxd,
                cyd,
                self._tmp_yfx,
                mfyd,
                n_split,
            )

        self._tracers_halo_updater.update(tracers.values())

        dp2 = self._tmp_dp

        for it in range(int(n_split)):
            last_call = it == n_split - 1
            self._dp_fluxadjustment(
                dp1,
                mfxd,
                mfyd,
                self.grid_data.rarea,
                dp2,
            )
            for q in tracers.values():
                self.finite_volume_transport(
                    self._copy_corners(q.storage),
                    cxd,
                    cyd,
                    self._tmp_xfx,
                    self._tmp_yfx,
                    self._tmp_fx,
                    self._tmp_fy,
                    x_mass_flux=mfxd,
                    y_mass_flux=mfyd,
                )
                self._q_adjust(
                    q.storage,
                    dp1,
                    self._tmp_fx,
                    self._tmp_fy,
                    self.grid_data.rarea,
                    dp2,
                )
            if not last_call:
                self._tracers_halo_updater.update(tracers.values())
                # use variable assignment to avoid a data copy
                dp1, dp2 = dp2, dp1
