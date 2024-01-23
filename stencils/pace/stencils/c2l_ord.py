from gt4py.cartesian.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import pace.dsl.gt4py_utils as utils
import pace.util
from pace import fv3core
from pace.dsl.dace.wrapped_halo_exchange import WrappedHaloUpdater
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import Float, FloatField, FloatFieldIJ
from pace.util.constants import X_DIM, X_INTERFACE_DIM, Y_DIM, Y_INTERFACE_DIM, Z_DIM
from pace.util.grid import GridData


A1 = 0.5625
A2 = -0.0625
C1 = 1.125
C2 = -0.125


def mock_exchange(
    quantity,
    domain_2d,
):
    isc = domain_2d[0][0]
    iec = domain_2d[0][1]
    isd = domain_2d[1][0]
    ied = domain_2d[1][1]
    jsc = domain_2d[2][0]
    jec = domain_2d[2][1]
    jsd = domain_2d[3][0]
    jed = domain_2d[3][1]
    nhalo = isc - isd

    quantity[isd:isc, :, :] = quantity[iec - nhalo + 1 : iec + 1, :, :]
    quantity[iec + 1 : ied + 1, :, :] = quantity[isc : isc + nhalo, :, :]
    quantity[:, jsd:jsc, :] = quantity[:, jec - nhalo + 1 : jec + 1, :]
    quantity[:, jec + 1 : jed + 1, :] = quantity[:, jsc : jsc + nhalo, :]

    quantity[isd:isc, jsd:jsc, :] = quantity[
        iec - nhalo + 1 : iec + 1, jec - nhalo + 1 : jec + 1, :
    ]
    quantity[isd:isc, jec + 1 : jed + 1, :] = quantity[
        iec - nhalo + 1 : iec + 1, jsc : jsc + nhalo, :
    ]
    quantity[iec + 1 : ied + 1, jsd:jsc, :] = quantity[
        isc : isc + nhalo, jec - nhalo + 1 : jec + 1, :
    ]
    quantity[iec + 1 : ied + 1, jec + 1 : jed + 1, :] = quantity[
        isc : isc + nhalo, jsc : jsc + nhalo, :
    ]


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
    """
    Args:
        u (in):
        v (in):
        dx (in):
        dy (in):
        a11 (in):
        a12 (in):
        a21 (in):
        a22 (in):
        ua (out):
        va (out):
    """
    from __externals__ import grid_type

    with computation(PARALLEL), interval(...):
        if __INLINED(grid_type < 4):
            wu = u * dx
            wv = v * dy
            # Co-variant vorticity-conserving interpolation
            u1 = 2.0 * (wu + wu[0, 1, 0]) / (dx + dx[0, 1])
            v1 = 2.0 * (wv + wv[1, 0, 0]) / (dy + dy[1, 0])
            # Cubed (cell center co-variant winds) to lat-lon
            ua = a11 * u1 + a12 * v1
            va = a21 * u1 + a22 * v1
        else:
            ua = 0.5 * (u + u[0, 1, 0])
            va = 0.5 * (v + v[1, 0, 0])


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
    """
    Args:
        u (in):
        v (in):
        dx (in):
        dy (in):
        a11 (in):
        a12 (in):
        a21 (in):
        a22 (in):
        ua (out):
        va (out):
    """
    with computation(PARALLEL), interval(...):
        from __externals__ import grid_type, i_end, i_start, j_end, j_start

        if __INLINED(grid_type < 4):
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
        else:
            ua = A2 * (u[0, -1, 0] + u[0, 2, 0]) + A1 * (u + u[0, 1, 0])
            va = A2 * (v[-1, 0, 0] + v[2, 0, 0]) + A1 * (v + v[1, 0, 0])


class CubedToLatLon:
    """
    Fortan name is c2l_ord2
    """

    def __init__(
        self,
        state: fv3core.DycoreState,
        stencil_factory: StencilFactory,
        quantity_factory: pace.util.QuantityFactory,
        grid_data: GridData,
        grid_type: int,
        order: int,
        comm: pace.util.Communicator,
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
        isc = grid_indexing.isc
        jsc = grid_indexing.jsc
        iec = grid_indexing.iec
        jec = grid_indexing.jec
        isd = grid_indexing.isd
        jsd = grid_indexing.jsd
        ied = grid_indexing.ied
        jed = grid_indexing.jed
        self._domain = [[isc, iec], [isd, ied], [jsc, jec], [jsd, jed]]

        self._n_halo = grid_indexing.n_halo
        self._dx = grid_data.dx
        self._dy = grid_data.dy
        if comm.size == 1:
            self.one_rank = True
        else:
            self.one_rank = False

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
            func=func,
            externals={"grid_type": grid_type},
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
            compute_halos=halos,
        )

        origin = grid_indexing.origin_compute()
        shape = grid_indexing.max_shape
        if not self.one_rank:
            full_size_xyiz_halo_spec = quantity_factory.get_quantity_halo_spec(
                dims=[X_DIM, Y_INTERFACE_DIM, Z_DIM],
                n_halo=grid_indexing.n_halo,
                dtype=Float,
            )
            full_size_xiyz_halo_spec = quantity_factory.get_quantity_halo_spec(
                dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM],
                n_halo=grid_indexing.n_halo,
                dtype=Float,
            )
            self.u__v = WrappedHaloUpdater(
                comm.get_vector_halo_updater(
                    [full_size_xyiz_halo_spec], [full_size_xiyz_halo_spec]
                ),
                state,
                ["u"],
                ["v"],
                comm=comm,
            )

    def __call__(
        self,
        u: FloatField,
        v: FloatField,
        ua: FloatField,
        va: FloatField,
    ):
        """
        Interpolate D-grid to A-grid winds at latitude-longitude coordinates.
        Args:
            u: x-wind on D-grid (in)
            v: y-wind on D-grid (in)
            ua: x-wind on A-grid (out)
            va: y-wind on A-grid (out)
            comm: Cubed-sphere or Tile communicator
        """
        if self._do_ord4:
            if self.one_rank:
                mock_exchange(u[:, :-1, :], self._domain)
                mock_exchange(v[:-1, :, :], self._domain)
            else:
                self.u__v.update()
        self._compute_cubed_to_latlon(
            u,
            v,
            self._dx,
            self._dy,
            self._a11,
            self._a12,
            self._a21,
            self._a22,
            ua,
            va,
        )
