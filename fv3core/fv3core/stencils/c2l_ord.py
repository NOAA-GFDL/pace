from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core.utils.gt4py_utils as utils
from fv3core.utils.grid import GridData
from fv3core.utils.stencil import StencilFactory
from fv3core.utils.typing import FloatField, FloatFieldIJ
from fv3gfs.util import CubedSphereCommunicator
from fv3gfs.util.constants import X_DIM, Y_DIM, Z_DIM
from fv3gfs.util.quantity import Quantity


C1 = 1.125
C2 = -0.125


@utils.mark_untested("This namelist option is not tested")
def c2l_ord2(
    u: FloatField,
    v: FloatField,
    dx: FloatFieldIJ,
    dy: FloatFieldIJ,
    a11: FloatFieldIJ,
    a12: FloatFieldIJ,
    a21: FloatFieldIJ,
    a22: FloatFieldIJ,
    ua: FloatField,
    va: FloatField,
):
    with computation(PARALLEL), interval(...):
        wu = u * dx
        wv = v * dy
        # Co-variant vorticity-conserving interpolation
        u1 = 2.0 * (wu + wu[0, 1, 0]) / (dx + dx[0, 1])
        v1 = 2.0 * (wv + wv[1, 0, 0]) / (dy + dy[1, 0])
        # Cubed (cell center co-variant winds) to lat-lon
        ua = a11 * u1 + a12 * v1
        va = a21 * u1 + a22 * v1


def ord4_transform(
    u: FloatField,
    v: FloatField,
    dx: FloatFieldIJ,
    dy: FloatFieldIJ,
    a11: FloatFieldIJ,
    a12: FloatFieldIJ,
    a21: FloatFieldIJ,
    a22: FloatFieldIJ,
    ua: FloatField,
    va: FloatField,
):
    with computation(PARALLEL), interval(...):
        from __externals__ import i_end, i_start, j_end, j_start

        utmp = C2 * (u[0, -1, 0] + u[0, 2, 0]) + C1 * (u + u[0, 1, 0])
        vtmp = C2 * (v[-1, 0, 0] + v[2, 0, 0]) + C1 * (v + v[1, 0, 0])

        # south/north edge
        with horizontal(region[:, j_start], region[:, j_end]):
            vtmp = 2.0 * ((v * dy) + (v[1, 0, 0] * dy[1, 0])) / (dy + dy[1, 0])
            utmp = 2.0 * (u * dx + u[0, 1, 0] * dx[0, 1]) / (dx + dx[0, 1])

        # west/east edge
        with horizontal(region[i_start, :], region[i_end, :]):
            utmp = 2.0 * ((u * dx) + (u[0, 1, 0] * dx[0, 1])) / (dx + dx[0, 1])
            vtmp = 2.0 * ((v * dy) + (v[1, 0, 0] * dy[1, 0])) / (dy + dy[1, 0])

        # Transform local a-grid winds into latitude-longitude coordinates
        ua = a11 * utmp + a12 * vtmp
        va = a21 * utmp + a22 * vtmp


class CubedToLatLon:
    """
    Fortan name is c2l_ord2
    """

    def __init__(
        self, stencil_factory: StencilFactory, grid_data: GridData, order: int
    ):
        """
        Initializes stencils to use either 2nd or 4th order of interpolation
        based on namelist setting
        Args:
            stencil_factory: creates stencils
            grid_data: object with metric terms
            order: Order of interpolation, must be 2 or 4
        """
        grid_indexing = stencil_factory.grid_indexing
        self._n_halo = grid_indexing.n_halo
        self._dx = grid_data.dx
        self._dy = grid_data.dy

        # TODO: maybe compute locally a* variables
        # They depend on z* and sin_sg5, which
        # currently aren't in GridData but are used
        # in other parts of the model and might be added.
        self._a11 = grid_data.a11
        self._a12 = grid_data.a12
        self._a21 = grid_data.a21
        self._a22 = grid_data.a22
        if order == 2:
            self._do_ord4 = False
            halos = (1, 1)
            func = c2l_ord2
        else:
            self._do_ord4 = True
            halos = (0, 0)
            func = ord4_transform
        self._compute_cubed_to_latlon = stencil_factory.from_dims_halo(
            func=func, compute_dims=[X_DIM, Y_DIM, Z_DIM], compute_halos=halos
        )

    def __call__(
        self,
        u: Quantity,
        v: Quantity,
        ua: FloatField,
        va: FloatField,
        comm: CubedSphereCommunicator,
    ):
        """
        Interpolate D-grid to A-grid winds at latitude-longitude coordinates.
        Args:
            u: x-wind on D-grid (in)
            v: y-wind on D-grid (in)
            ua: x-wind on A-grid (out)
            va: y-wind on A-grid (out)
            comm: Cubed-sphere communicator
        """
        if self._do_ord4:
            comm.vector_halo_update(u, v, n_points=self._n_halo)
        self._compute_cubed_to_latlon(
            u.storage,
            v.storage,
            self._dx,
            self._dy,
            self._a11,
            self._a12,
            self._a21,
            self._a22,
            ua,
            va,
        )
