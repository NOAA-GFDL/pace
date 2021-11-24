import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    __INLINED,
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    horizontal,
    interval,
    log,
    region,
    sin,
)

import fv3core.utils.global_constants as constants
from fv3gfs.util import X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM
from fv3gfs.util.pace.grid import axis_offsets
from fv3gfs.util.pace.stencil import StencilFactory
from fv3gfs.util.pace.typing import FloatField, FloatFieldK


SDAY = 86400.0

# NOTE: The fortran version of this computes rf in the first timestep only. Then
# rf_initialized let's you know you can skip it. Here we calculate it every
# time.
@gtscript.function
def compute_rf_vals(pfull, bdt, rf_cutoff, tau0, ptop):
    return (
        bdt
        / tau0
        * sin(0.5 * constants.PI * log(rf_cutoff / pfull) / log(rf_cutoff / ptop)) ** 2
    )


@gtscript.function
def compute_rff_vals(pfull, dt, rf_cutoff, tau0, ptop):
    rffvals = compute_rf_vals(pfull, dt, rf_cutoff, tau0, ptop)
    rffvals = 1.0 / (1.0 + rffvals)
    return rffvals


@gtscript.function
def dm_layer(rf, dp, wind):
    return (1.0 - rf) * dp * wind


def ray_fast_wind_compute(
    u: FloatField,
    v: FloatField,
    w: FloatField,
    dp: FloatFieldK,
    pfull: FloatFieldK,
    dt: float,
    ptop: float,
    rf_cutoff_nudge: float,
    ks: int,
):
    from __externals__ import hydrostatic, local_ie, local_je, rf_cutoff, tau

    # dm_stencil
    with computation(PARALLEL), interval(...):
        # TODO -- in the fortran model rf is only computed once, repeating
        # the computation every time ray_fast is run is inefficient
        if pfull < rf_cutoff:
            rf = compute_rff_vals(pfull, dt, rf_cutoff, tau * SDAY, ptop)
    with computation(FORWARD):
        with interval(0, 1):
            if pfull < rf_cutoff_nudge:  # TODO and kaxes(k) < ks:
                dm = dp
        with interval(1, None):
            dm = dm[0, 0, -1]
            if pfull < rf_cutoff_nudge:  # TODO and kaxes(k) < ks:
                dm += dp
    with computation(BACKWARD), interval(0, -1):
        if pfull < rf_cutoff_nudge:
            dm = dm[0, 0, 1]
    # ray_fast_wind(u)
    with computation(FORWARD):
        with interval(0, 1):
            with horizontal(region[: local_ie + 1, :]):
                if pfull < rf_cutoff:
                    dmdir = dm_layer(rf, dp, u)
                    u *= rf
                else:
                    dm = 0
        with interval(1, None):
            with horizontal(region[: local_ie + 1, :]):
                dmdir = dmdir[0, 0, -1]
                if pfull < rf_cutoff:
                    dmdir += dm_layer(rf, dp, u)
                    u *= rf
    with computation(BACKWARD), interval(0, -1):
        if pfull < rf_cutoff:
            dmdir = dmdir[0, 0, 1]
    with computation(PARALLEL), interval(...):
        with horizontal(region[: local_ie + 1, :]):
            if pfull < rf_cutoff_nudge:  # TODO and axes(k) < ks:
                u += dmdir / dm
    # ray_fast_wind(v)
    with computation(FORWARD):
        with interval(0, 1):
            with horizontal(region[:, : local_je + 1]):
                if pfull < rf_cutoff:
                    dmdir = dm_layer(rf, dp, v)
                    v *= rf
                else:
                    dm = 0
        with interval(1, None):
            with horizontal(region[:, : local_je + 1]):
                dmdir = dmdir[0, 0, -1]
                if pfull < rf_cutoff:
                    dmdir += dm_layer(rf, dp, v)
                    v *= rf
    with computation(BACKWARD), interval(0, -1):
        if pfull < rf_cutoff:
            dmdir = dmdir[0, 0, 1]
    with computation(PARALLEL), interval(...):
        with horizontal(region[:, : local_je + 1]):
            if pfull < rf_cutoff_nudge:  # TODO and axes(k) < ks:
                v += dmdir / dm
    # ray_fast_w
    with computation(PARALLEL), interval(...):
        with horizontal(region[: local_ie + 1, : local_je + 1]):
            if __INLINED(not hydrostatic):
                if pfull < rf_cutoff:
                    w *= rf


class RayleighDamping:
    """
    Apply Rayleigh damping (for tau > 0).

    Namelist:
        - tau [Float]: time scale (in days) for Rayleigh friction applied to horizontal
                       and vertical winds; lost kinetic energy is converted to heat,
                       except on nested grids.
        - rf_cutoff [Float]: pressure below which no Rayleigh damping is applied
                             if tau > 0.

    Fotran name: ray_fast.
    """

    def __init__(self, stencil_factory: StencilFactory, rf_cutoff, tau, hydrostatic):
        grid_indexing = stencil_factory.grid_indexing
        self._rf_cutoff = rf_cutoff
        origin, domain = grid_indexing.get_origin_domain(
            [X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM]
        )

        ax_offsets = axis_offsets(grid_indexing, origin, domain)
        local_axis_offsets = {}
        for axis_offset_name, axis_offset_value in ax_offsets.items():
            if "local" in axis_offset_name:
                local_axis_offsets[axis_offset_name] = axis_offset_value

        self._ray_fast_wind_compute = stencil_factory.from_origin_domain(
            ray_fast_wind_compute,
            origin=origin,
            domain=domain,
            externals={
                "hydrostatic": hydrostatic,
                "rf_cutoff": rf_cutoff,
                "tau": tau,
                **local_axis_offsets,
            },
        )

    def __call__(
        self,
        u: FloatField,
        v: FloatField,
        w: FloatField,
        dp: FloatFieldK,
        pfull: FloatFieldK,
        dt: float,
        ptop: float,
        ks: int,
    ):
        rf_cutoff_nudge = self._rf_cutoff + min(100.0, 10.0 * ptop)

        self._ray_fast_wind_compute(
            u,
            v,
            w,
            dp,
            pfull,
            dt,
            ptop,
            rf_cutoff_nudge,
            ks,
        )
