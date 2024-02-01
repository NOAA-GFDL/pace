from gt4py.cartesian.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)
from ndsl.dsl.dace import orchestrate
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from ndsl.grid import GridData

from pace.fv3core.stencils.d2a2c_vect import contravariant


def main_uc_vc_contra(
    uc: FloatField,
    vc: FloatField,
    cosa_u: FloatFieldIJ,
    rsin_u: FloatFieldIJ,
    cosa_v: FloatFieldIJ,
    rsin_v: FloatFieldIJ,
    uc_contra: FloatField,
    vc_contra: FloatField,
):
    """
    Args:
        uc (in): covariant c-grid x-wind
        vc (in): covariant c-grid y-wind
        cosa_u (in): ???
        rsin_u (in): ???
        cosa_v (in): ???
        rsin_v (in): ???
        uc_contra (out): contravariant c-grid x-wind
        vc_contra (out): contravariant c-grid y-wind
    """
    from __externals__ import (
        grid_type,
        j_end,
        j_start,
        local_ie,
        local_is,
        local_je,
        local_js,
    )

    with computation(PARALLEL), interval(...):
        if __INLINED(grid_type < 3):
            utmp = uc_contra
            with horizontal(region[local_is - 1 : local_ie + 3, :]):
                # for C-grid, v must be regridded to lie at the same point as u
                v = 0.25 * (vc[-1, 0, 0] + vc + vc[-1, 1, 0] + vc[0, 1, 0])
                uc_contra = contravariant(uc, v, cosa_u, rsin_u)
            # TODO: investigate whether this region operation is necessary
            with horizontal(
                region[:, j_start - 1 : j_start + 1], region[:, j_end : j_end + 2]
            ):
                uc_contra = utmp

            with horizontal(region[:, local_js - 1 : local_je + 3]):
                # for C-grid, u must be regridded to lie at same point as v
                u = 0.25 * (uc[0, -1, 0] + uc[1, -1, 0] + uc + uc[1, 0, 0])
                vc_contra = contravariant(vc, u, cosa_v, rsin_v)
        else:
            uc_contra = uc
            vc_contra = vc


def uc_contra_y_edge(
    uc: FloatField,
    sin_sg1: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    uc_contra: FloatField,
):
    """
    On the edges, the definition of contravariant C-grid wind is ambiguous,
    so we use the average normal vector as the new normal vector.

    This uses the upstream value of the metric term sin_sg (on both sides)
    instead of some kind of average to modify the rsin2 term used in the
    covariant -> contravariant conversion.

    ucontra = uc / sin(alpha) at the edges

    Args:
        uc (in):
        sin_sg1 (in):
        sin_sg3 (in):
        uc_contra (out):
    """
    from __externals__ import i_end, i_start

    with computation(PARALLEL), interval(...):
        with horizontal(region[i_start, :], region[i_end + 1, :]):
            uc_contra = (uc / sin_sg3[-1, 0]) if (uc > 0) else (uc / sin_sg1)


def vc_contra_y_edge(
    vc: FloatField, cosa_v: FloatFieldIJ, uc_contra: FloatField, vc_contra: FloatField
):
    """
    Args:
        vc (in):
        cosa_v (in):
        uc_contra (in):
        vc_contra (inout):
    """
    from __externals__ import i_end, i_start, j_end, j_start, local_je, local_js

    # This works for 6 ranks, but not 54:
    # with horizontal(region[i_start - 1: i_start + 1, j_start + 2:j_end], \
    #                region[i_end : i_end + 2, j_start+2:j_end]):
    #    vt = vc - 0.25 * cosa_v * (
    #        ut[0, -1, 0] + ut[1, -1, 0] + ut + ut[1, 0, 0]
    #    )
    # original bounds with stencil calls
    # j1 = grid().js + 2 if grid().south_edge else grid().js
    # j2 = grid().je if grid().north_edge else grid().je + 2
    # TODO: this is a hack, copying vt to vtmp to 'correct' the edges
    # Can we *just* apply edge calculations in the correct regions without overcomputing
    # rank 0, 1, 2: local_js + 2:local_je + 2
    # rank 3, 4, 5: local_js:local_je + 2
    # rank 6, 7, 8: local_js:local_je
    with computation(PARALLEL), interval(...):
        vtmp = vc_contra
        with horizontal(
            region[i_start - 1 : i_start + 1, local_js : local_je + 2],
            region[i_end : i_end + 2, local_js : local_je + 2],
        ):
            u_contra = 0.25 * (
                uc_contra[0, -1, 0]
                + uc_contra[1, -1, 0]
                + uc_contra
                + uc_contra[1, 0, 0]
            )
            vc_contra = contravariant(vc, u_contra, cosa_v, 1.0)
        with horizontal(
            region[i_start - 1 : i_start + 1, j_start : j_start + 2],
            region[i_end : i_end + 2, j_start : j_start + 2],
            region[i_start - 1 : i_start + 1, j_end : j_end + 2],
            region[i_end : i_end + 2, j_end : j_end + 2],
        ):
            vc_contra = vtmp


def vc_contra_x_edge(
    vc: FloatField,
    sin_sg2: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    vc_contra: FloatField,
):
    """
    Args:
        vc (in):
        sin_sg2 (in):
        sin_sg4 (in):
        vc_contra (out):
    """
    from __externals__ import j_end, j_start

    with computation(PARALLEL), interval(...):
        with horizontal(region[:, j_start], region[:, j_end + 1]):
            vc_contra = (vc / sin_sg4[0, -1]) if (vc > 0) else (vc / sin_sg2)


def uc_contra_x_edge(
    uc: FloatField, cosa_u: FloatFieldIJ, vc_contra: FloatField, uc_contra: FloatField
):
    """
    Args:
        uc (in):
        cosa_u (in):
        vc_contra (in):
        uc_contra (inout):
    """
    from __externals__ import i_end, i_start, j_end, j_start, local_ie, local_is

    with computation(PARALLEL), interval(...):
        # TODO: parallel to what done for the vt_y_edge section
        utmp = uc_contra
        with horizontal(
            region[local_is : local_ie + 2, j_start - 1 : j_start + 1],
            region[local_is : local_ie + 2, j_end : j_end + 2],
        ):
            v_contra = 0.25 * (
                vc_contra[-1, 0, 0]
                + vc_contra
                + vc_contra[-1, 1, 0]
                + vc_contra[0, 1, 0]
            )
            uc_contra = contravariant(uc, v_contra, cosa_u, 1.0)
        with horizontal(
            region[i_start : i_start + 2, j_start - 1 : j_start + 1],
            region[i_start : i_start + 2, j_end : j_end + 2],
            region[i_end : i_end + 2, j_start - 1 : j_start + 1],
            region[i_end : i_end + 2, j_end : j_end + 2],
        ):
            uc_contra = utmp


def uc_contra_corners(
    cosa_u: FloatFieldIJ,
    cosa_v: FloatFieldIJ,
    uc: FloatField,
    vc: FloatField,
    uc_contra: FloatField,
    uc_contra_copy: FloatField,
    vc_contra: FloatField,
):
    """
    The following code (and vt_corners) solves a 2x2 system to
    get the interior parallel-to-edge uc,vc values near the corners
    (ex: for the sw corner ut(2,1) and vt(1,2) are solved for simultaneously).
    It then computes the halo uc, vc values so as to be consistent with the
    computations on the facing panel.

    The system solved is:
        ut(2,1) = uc(2,1) - avg(vt)*cosa_u(2,1)
        vt(1,2) = vc(1,2) - avg(ut)*cosa_v(1,2)
        in which avg(vt) includes vt(1,2) and avg(ut) includes ut(2,1)

    Args:
        cosa_u (in):
        cosa_v (in):
        uc (in):
        vc (in):
        uc_contra (out):
        uc_contra_copy (in): the covariant u-wind on ??? grid
        vc_contra (in):
    """
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v[-1, 0])
        with horizontal(region[i_start + 1, j_start - 1], region[i_start + 1, j_end]):
            # we can derive why there's a 0.25 and not 1/3rd factor below, based on the
            # system being solved as documented above
            # use Kramer's rule to solve the 2x2 matrix instead of Gaussian elimination
            # we verified the math works out.
            uc_contra = (
                uc
                - 0.25
                * cosa_u
                * (
                    vc_contra[-1, 1, 0]
                    + vc_contra[0, 1, 0]
                    + vc_contra
                    + vc[-1, 0, 0]
                    - 0.25
                    * cosa_v[-1, 0]
                    * (
                        uc_contra_copy[-1, 0, 0]
                        + uc_contra_copy[-1, -1, 0]
                        + uc_contra_copy[0, -1, 0]
                    )
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v[-1, 1])
        with horizontal(region[i_start + 1, j_start], region[i_start + 1, j_end + 1]):
            damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v[-1, 1])
            uc_contra = (
                uc
                - 0.25
                * cosa_u
                * (
                    vc_contra[-1, 0, 0]
                    + vc_contra
                    + vc_contra[0, 1, 0]
                    + vc[-1, 1, 0]
                    - 0.25
                    * cosa_v[-1, 1]
                    * (
                        uc_contra_copy[-1, 0, 0]
                        + uc_contra_copy[-1, 1, 0]
                        + uc_contra_copy[0, 1, 0]
                    )
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v)
        with horizontal(region[i_end, j_start - 1], region[i_end, j_end]):
            uc_contra = (
                uc
                - 0.25
                * cosa_u
                * (
                    vc_contra[0, 1, 0]
                    + vc_contra[-1, 1, 0]
                    + vc_contra[-1, 0, 0]
                    + vc
                    - 0.25
                    * cosa_v
                    * (
                        uc_contra_copy[1, 0, 0]
                        + uc_contra_copy[1, -1, 0]
                        + uc_contra_copy[0, -1, 0]
                    )
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v[0, 1])
        with horizontal(region[i_end, j_start], region[i_end, j_end + 1]):
            uc_contra = (
                uc
                - 0.25
                * cosa_u
                * (
                    vc_contra
                    + vc_contra[-1, 0, 0]
                    + vc_contra[-1, 1, 0]
                    + vc[0, 1, 0]
                    - 0.25
                    * cosa_v[0, 1]
                    * (
                        uc_contra_copy[1, 0, 0]
                        + uc_contra_copy[1, 1, 0]
                        + uc_contra_copy[0, 1, 0]
                    )
                )
            ) * damp


def vc_contra_corners(
    cosa_u: FloatFieldIJ,
    cosa_v: FloatFieldIJ,
    uc: FloatField,
    vc: FloatField,
    ut: FloatField,
    vc_contra: FloatField,
    vc_contra_copy: FloatField,
):
    """
    Args:
        cosa_u (in):
        cosa_v (in):
        uc (in):
        vc (in):
        ut (in):
        vc_contra (out):
        vc_contra_copy (in): the covariant v-wind on ??? grid
    """
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        damp = 1.0 / (1.0 - 0.0625 * cosa_u[0, -1] * cosa_v)
        with horizontal(region[i_start - 1, j_start + 1], region[i_end, j_start + 1]):
            vc_contra = (
                vc
                - 0.25
                * cosa_v
                * (
                    ut[1, -1, 0]
                    + ut[1, 0, 0]
                    + ut
                    + uc[0, -1, 0]
                    - 0.25
                    * cosa_u[0, -1]
                    * (
                        vc_contra_copy[0, -1, 0]
                        + vc_contra_copy[-1, -1, 0]
                        + vc_contra_copy[-1, 0, 0]
                    )
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u[1, -1] * cosa_v)
        with horizontal(region[i_start, j_start + 1], region[i_end + 1, j_start + 1]):
            vc_contra = (
                vc
                - 0.25
                * cosa_v
                * (
                    ut[0, -1, 0]
                    + ut
                    + ut[1, 0, 0]
                    + uc[1, -1, 0]
                    - 0.25
                    * cosa_u[1, -1]
                    * (
                        vc_contra_copy[0, -1, 0]
                        + vc_contra_copy[1, -1, 0]
                        + vc_contra_copy[1, 0, 0]
                    )
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u[1, 0] * cosa_v)
        with horizontal(region[i_end + 1, j_end], region[i_start, j_end]):
            vc_contra = (
                vc
                - 0.25
                * cosa_v
                * (
                    ut
                    + ut[0, -1, 0]
                    + ut[1, -1, 0]
                    + uc[1, 0, 0]
                    - 0.25
                    * cosa_u[1, 0]
                    * (
                        vc_contra_copy[0, 1, 0]
                        + vc_contra_copy[1, 1, 0]
                        + vc_contra_copy[1, 0, 0]
                    )
                )
            ) * damp
        damp = 1.0 / (1.0 - 0.0625 * cosa_u * cosa_v)
        with horizontal(region[i_end, j_end], region[i_start - 1, j_end]):
            vc_contra = (
                vc
                - 0.25
                * cosa_v
                * (
                    ut[1, 0, 0]
                    + ut[1, -1, 0]
                    + ut[0, -1, 0]
                    + uc
                    - 0.25
                    * cosa_u
                    * (
                        vc_contra_copy[0, 1, 0]
                        + vc_contra_copy[-1, 1, 0]
                        + vc_contra_copy[-1, 0, 0]
                    )
                )
            ) * damp


"""
# Single stencil version to use when possible with gt backends
def fxadv_stencil(
    cosa_u: FloatFieldIJ,
    cosa_v: FloatFieldIJ,
    rsin_u: FloatFieldIJ,
    rsin_v: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    uc: FloatField,
    vc: FloatField,
    ut: FloatField,
    vt: FloatField,
    dt: Float,
):
    with computation(PARALLEL), interval(...):
        ut = main_ut(uc, vc, cosa_u, rsin_u, ut)
        ut = ut_y_edge(uc, sin_sg1, sin_sg3, ut, dt)
        vt = main_vt(uc, vc, cosa_v, rsin_v, vt)
        vt = vt_y_edge(vc, cosa_v, ut, vt)
        vt = vt_x_edge(vc, sin_sg2, sin_sg4, vt, dt)
        ut = ut_x_edge(uc, cosa_u, vt, ut)
        ut = ut_corners(uc, vc, cosa_u, cosa_v, ut, vt)
        vt = vt_corners(uc, vc, cosa_u, cosa_v, ut, vt)
"""


def fxadv_fluxes_stencil(
    sin_sg1: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    rdxa: FloatFieldIJ,
    rdya: FloatFieldIJ,
    dy: FloatFieldIJ,
    dx: FloatFieldIJ,
    crx: FloatField,
    cry: FloatField,
    x_area_flux: FloatField,
    y_area_flux: FloatField,
    uc_contra: FloatField,
    vc_contra: FloatField,
    dt: Float,
):
    """
    Args:
        sin_sg1 (in):
        sin_sg2 (in):
        sin_sg3 (in):
        sin_sg4 (in):
        rdxa (in):
        rdya (in):
        dy (in):
        dx (in):
        crx (out):
        cry (out):
        x_area_flux (out):
        y_area_flux (out):
        uc_contra (in):
        vc_contra (in):
    """
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):
        with horizontal(region[local_is : local_ie + 2, :]):
            if uc_contra > 0:
                crx = dt * uc_contra * rdxa[-1, 0]
                x_area_flux = dy * dt * uc_contra * sin_sg3[-1, 0]
            else:
                crx = dt * uc_contra * rdxa
                x_area_flux = dy * dt * uc_contra * sin_sg1
        with horizontal(region[:, local_js : local_je + 2]):
            if vc_contra > 0:
                cry = dt * vc_contra * rdya[0, -1]
                y_area_flux = dx * dt * vc_contra * sin_sg4[0, -1]
            else:
                cry = dt * vc_contra * rdya
                y_area_flux = dx * dt * vc_contra * sin_sg2


class FiniteVolumeFluxPrep:
    """
    A large section of code near the beginning of Fortran's d_sw subroutinw
    Known in this repo as FxAdv,
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid_data: GridData,
        grid_type: int,
    ):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
        )
        grid_indexing = stencil_factory.grid_indexing
        self._grid_type = grid_type
        self._tile_interior = not (
            grid_indexing.west_edge
            or grid_indexing.east_edge
            or grid_indexing.north_edge
            or grid_indexing.south_edge
        )
        self._dx = grid_data.dx
        self._dy = grid_data.dy
        self._rdxa = grid_data.rdxa
        self._rdya = grid_data.rdya
        self._cosa_u = grid_data.cosa_u
        self._cosa_v = grid_data.cosa_v
        self._rsin_u = grid_data.rsin_u
        self._rsin_v = grid_data.rsin_v
        self._sin_sg1 = grid_data.sin_sg1
        self._sin_sg2 = grid_data.sin_sg2
        self._sin_sg3 = grid_data.sin_sg3
        self._sin_sg4 = grid_data.sin_sg4
        origin = grid_indexing.origin_full()
        domain = grid_indexing.domain_full()
        ax_offsets = grid_indexing.axis_offsets(origin, domain)
        kwargs = {"externals": ax_offsets, "origin": origin, "domain": domain}
        origin_corners = grid_indexing.origin_full(add=(1, 1, 0))
        domain_corners = grid_indexing.domain_full(add=(-1, -1, 0))
        corner_offsets = grid_indexing.axis_offsets(origin_corners, domain_corners)
        kwargs_corners = {
            "externals": corner_offsets,
            "origin": origin_corners,
            "domain": domain_corners,
        }
        self._main_uc_vc_contra_stencil = stencil_factory.from_origin_domain(
            main_uc_vc_contra,
            externals={"grid_type": grid_type, **ax_offsets},
            origin=origin,
            domain=domain,
        )
        if self._grid_type < 3:
            self._uc_contra_y_edge_stencil = stencil_factory.from_origin_domain(
                uc_contra_y_edge, **kwargs
            )
            self._vc_contra_y_edge_stencil = stencil_factory.from_origin_domain(
                vc_contra_y_edge, **kwargs
            )
            self._vc_contra_x_edge_stencil = stencil_factory.from_origin_domain(
                vc_contra_x_edge, **kwargs
            )
            self._uc_contra_x_edge_stencil = stencil_factory.from_origin_domain(
                uc_contra_x_edge, **kwargs
            )
            self._uc_contra_corners_stencil = stencil_factory.from_origin_domain(
                uc_contra_corners, **kwargs_corners
            )
            self._vc_contra_corners_stencil = stencil_factory.from_origin_domain(
                vc_contra_corners, **kwargs_corners
            )
        self._fxadv_fluxes_stencil = stencil_factory.from_origin_domain(
            fxadv_fluxes_stencil, **kwargs
        )
        # self._set_nans = get_set_nan_func(
        #     grid_indexing,
        #     dims=[X_DIM, Y_DIM, Z_DIM],
        #     n_halo=((2, 2), (2, 2)),
        # )

    def __call__(
        self,
        uc,
        vc,
        crx,
        cry,
        x_area_flux,
        y_area_flux,
        uc_contra,
        vc_contra,
        dt,
    ):
        """
        Updates flux operators and courant numbers for fvtp2d
        To start off D_SW after the C-grid winds have been advanced half a timestep,
        and and compute finite volume transport on the D-grid (e.g.Putman and Lin 2007),
        this module prepares terms such as parts of equations 7 and 13 in Putnam and
        Lin, 2007, that get consumed by fvtp2d and ppm methods.

        Described in sections 3.2 and 4.2 of the FV3 documentation.

        Args:
            uc (in): covariant x-velocity on the C-grid
            vc (in): covariant y-velocity on the C-grid
            crx (out): Courant number, x direction
            cry (out): Courant number, y direction
            x_area_flux (out): flux of area in x-direction, in units of m^2
            y_area_flux (out): flux of area in y-direction, in units of m^2
            uc_contra (out): contravariant x-velocity on C-grid
            vc_contra (out): contravariant y-velocity on C-grid
            dt (in): acoustic timestep in seconds
        """

        # in the original Fortran code, uc_contra is named ut and vc_contra is vt

        self._main_uc_vc_contra_stencil(
            uc,
            vc,
            self._cosa_u,
            self._rsin_u,
            self._cosa_v,
            self._rsin_v,
            uc_contra,
            vc_contra,
        )
        if self._grid_type < 3:
            if not self._tile_interior:
                self._uc_contra_y_edge_stencil(
                    uc, self._sin_sg1, self._sin_sg3, uc_contra
                )
                self._vc_contra_y_edge_stencil(
                    vc,
                    self._cosa_v,
                    uc_contra,
                    vc_contra,
                )
                self._vc_contra_x_edge_stencil(
                    vc, self._sin_sg2, self._sin_sg4, vc_contra
                )
                self._uc_contra_x_edge_stencil(
                    uc,
                    self._cosa_u,
                    vc_contra,
                    uc_contra,
                )
                # NOTE: this is aliasing memory
                self._uc_contra_corners_stencil(
                    self._cosa_u,
                    self._cosa_v,
                    uc,
                    vc,
                    uc_contra,
                    uc_contra,
                    vc_contra,
                )
                # NOTE: this is aliasing memory
                self._vc_contra_corners_stencil(
                    self._cosa_u,
                    self._cosa_v,
                    uc,
                    vc,
                    uc_contra,
                    vc_contra,
                    vc_contra,
                )
        self._fxadv_fluxes_stencil(
            self._sin_sg1,
            self._sin_sg2,
            self._sin_sg3,
            self._sin_sg4,
            self._rdxa,
            self._rdya,
            self._dy,
            self._dx,
            crx,
            cry,
            x_area_flux,
            y_area_flux,
            uc_contra,
            vc_contra,
            dt,
        )
