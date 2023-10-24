import dataclasses
import functools
import warnings
from typing import Tuple

import numpy as np

from pace import util
from pace.dsl.gt4py_utils import asarray
from pace.dsl.stencil import GridIndexing
from pace.dsl.typing import Float
from pace.stencils.corners import (
    fill_corners_2d,
    fill_corners_agrid,
    fill_corners_cgrid,
    fill_corners_dgrid,
)
from pace.util import X_DIM, X_INTERFACE_DIM, Y_DIM, Y_INTERFACE_DIM, Z_INTERFACE_DIM
from pace.util.constants import N_HALO_DEFAULT, PI, RADIUS

from .eta import set_hybrid_pressure_coefficients
from .geometry import (
    calc_unit_vector_south,
    calc_unit_vector_west,
    calculate_divg_del6,
    calculate_grid_a,
    calculate_grid_z,
    calculate_l2c_vu,
    calculate_supergrid_cos_sin,
    calculate_trig_uv,
    calculate_xy_unit_vectors,
    edge_factors,
    efactor_a2c_v,
    get_center_vector,
    supergrid_corner_fix,
    unit_vector_lonlat,
)
from .gnomonic import (
    get_area,
    great_circle_distance_along_axis,
    local_gnomonic_ed,
    lon_lat_corner_to_cell_center,
    lon_lat_midpoint,
    lon_lat_to_xyz,
    set_c_grid_tile_border_area,
    set_corner_area_to_triangle_area,
    set_tile_border_dxc,
    set_tile_border_dyc,
)
from .mirror import mirror_grid


# TODO: when every environment in python3.8, remove
# this custom decorator
def cached_property(func):
    @property
    @functools.lru_cache()
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    return wrapper


def ignore_zero_division(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return wrapped


def quantity_cast_to_model_float(
    quantity_factory: util.QuantityFactory, qty_64: util.Quantity
) -> util.Quantity:
    """Copy & cast from 64-bit float to model precision if need be"""
    qty = quantity_factory.zeros(qty_64.dims, qty_64.units, dtype=Float)
    qty.data[:] = qty_64.data[:]
    return qty


@dataclasses.dataclass
class GridDefinition:
    dims: Tuple[str, ...]
    units: str


class GridDefinitions:
    CELL_CENTER = (X_DIM, Y_DIM)
    CELL_CORNERS = (X_INTERFACE_DIM, Y_INTERFACE_DIM)
    LON_OR_LAT_DIM = "lon_or_lat"
    TILE_DIM = "tile"
    CARTESIAN_DIM = "xyz_direction"

    grid = GridDefinition(dims=CELL_CORNERS + (LON_OR_LAT_DIM,), units="radians")
    agrid = GridDefinition(dims=CELL_CENTER + (LON_OR_LAT_DIM,), units="radians")
    lon = GridDefinition(dims=CELL_CORNERS, units="radians")
    lat = GridDefinition(dims=CELL_CORNERS, units="radians")
    lon_agrid = GridDefinition(dims=CELL_CENTER, units="radians")
    lat_agrid = GridDefinition(dims=CELL_CENTER, units="radians")
    area = GridDefinition(dims=CELL_CENTER, units="m^2")
    area_cgrid = GridDefinition(dims=CELL_CORNERS, units="m^2")
    rarea = GridDefinition(dims=area.dims, units="1/m^2")
    rarea_c = GridDefinition(dims=area_cgrid.dims, units="1/m^2")
    dx = GridDefinition(dims=(X_DIM, Y_INTERFACE_DIM), units="m")
    dy = GridDefinition(dims=(X_INTERFACE_DIM, Y_DIM), units="m")
    dxc = GridDefinition(dims=(X_INTERFACE_DIM, Y_DIM), units="m")
    dyc = GridDefinition(dims=(X_DIM, Y_INTERFACE_DIM), units="m")
    dxa = GridDefinition(dims=CELL_CENTER, units="m")
    dya = GridDefinition(dims=CELL_CENTER, units="m")
    rdx = GridDefinition(dims=dx.dims, units="1/m")
    rdy = GridDefinition(dims=dy.dims, units="1/m")
    rdxc = GridDefinition(dims=dxc.dims, units="1/m")
    rdyc = GridDefinition(dims=dyc.dims, units="1/m")
    rdxa = GridDefinition(dims=dxa.dims, units="1/m")
    rdya = GridDefinition(dims=dya.dims, units="1/m")
    ak = GridDefinition(dims=(Z_INTERFACE_DIM,), units="m")
    bk = GridDefinition(dims=(Z_INTERFACE_DIM,), units="m")
    ec1 = GridDefinition(dims=CELL_CENTER + (CARTESIAN_DIM,), units="m")
    ec2 = GridDefinition(dims=CELL_CENTER + (CARTESIAN_DIM,), units="m")
    ew1 = GridDefinition(dims=CELL_CORNERS + (CARTESIAN_DIM,), units="m")
    ew2 = GridDefinition(dims=CELL_CORNERS + (CARTESIAN_DIM,), units="m")
    es1 = GridDefinition(
        dims=(
            X_DIM,
            Y_INTERFACE_DIM,
            CARTESIAN_DIM,
        ),
        units="m",
    )
    es2 = GridDefinition(
        dims=(
            X_DIM,
            Y_INTERFACE_DIM,
            CARTESIAN_DIM,
        ),
        units="m",
    )
    cosa_u = GridDefinition(dims=(X_INTERFACE_DIM, Y_DIM), units="")
    cosa_v = GridDefinition(dims=(X_DIM, Y_INTERFACE_DIM), units="")
    cosa_s = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    sina_u = GridDefinition(dims=(X_INTERFACE_DIM, Y_DIM), units="")
    sina_v = GridDefinition(dims=(X_DIM, Y_INTERFACE_DIM), units="")
    rsin_u = GridDefinition(dims=(X_INTERFACE_DIM, Y_DIM), units="")
    rsin_v = GridDefinition(dims=(X_DIM, Y_INTERFACE_DIM), units="")
    rsina = GridDefinition(dims=(X_INTERFACE_DIM, Y_INTERFACE_DIM), units="")
    rsin2 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    cosa = GridDefinition(dims=(X_INTERFACE_DIM, Y_INTERFACE_DIM), units="")
    sina = GridDefinition(dims=(X_INTERFACE_DIM, Y_INTERFACE_DIM), units="")
    cos_sg1 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    cos_sg2 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    cos_sg3 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    cos_sg4 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    cos_sg5 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    cos_sg6 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    cos_sg7 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    cos_sg8 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    cos_sg9 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    sin_sg1 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    sin_sg2 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    sin_sg3 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    sin_sg4 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    sin_sg5 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    sin_sg6 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    sin_sg7 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    sin_sg8 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    sin_sg9 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    l2c_u = GridDefinition(dims=(X_DIM, Y_INTERFACE_DIM), units="")
    l2c_v = GridDefinition(dims=(X_INTERFACE_DIM, Y_DIM), units="")
    ee1 = GridDefinition(dims=CELL_CORNERS + (CARTESIAN_DIM,), units="")
    ee2 = GridDefinition(dims=CELL_CORNERS + (CARTESIAN_DIM,), units="")
    del6_u = GridDefinition(dims=(X_DIM, Y_INTERFACE_DIM), units="")
    del6_v = GridDefinition(dims=(X_INTERFACE_DIM, Y_DIM), units="")
    divg_u = GridDefinition(dims=(X_DIM, Y_INTERFACE_DIM), units="")
    divg_v = GridDefinition(dims=(X_INTERFACE_DIM, Y_DIM), units="")
    z11 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    z12 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    z21 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    z22 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    a11 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    a12 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    a21 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    a22 = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    edge_s = GridDefinition(dims=(X_INTERFACE_DIM,), units="")
    edge_n = GridDefinition(dims=(X_INTERFACE_DIM,), units="")
    edge_e = GridDefinition(
        dims=(
            X_DIM,
            Y_INTERFACE_DIM,
        ),
        units="",
    )
    edge_w = GridDefinition(
        dims=(
            X_DIM,
            Y_INTERFACE_DIM,
        ),
        units="",
    )
    edge_vect_s = GridDefinition(dims=(X_DIM,), units="")
    edge_vect_n = GridDefinition(dims=(X_DIM,), units="")
    edge_vect_e_1d = GridDefinition(dims=(Y_DIM,), units="")
    edge_vect_w_1d = GridDefinition(dims=(Y_DIM,), units="")
    edge_vect_e = GridDefinition(dims=(X_DIM, Y_DIM), units="")
    edge_vect_w = GridDefinition(dims=(X_DIM, Y_DIM), units="")


# TODO
# corners use sizer + partitioner rather than GridIndexer,
# have to refactor fv3core calls to corners to do this as well
class MetricTerms:
    LON_OR_LAT_DIM = GridDefinitions.LON_OR_LAT_DIM
    TILE_DIM = GridDefinitions.TILE_DIM
    CARTESIAN_DIM = GridDefinitions.CARTESIAN_DIM
    N_TILES = 6
    RIGHT_HAND_GRID = False

    def __init__(
        self,
        *,
        quantity_factory: util.QuantityFactory,
        communicator: util.Communicator,
        grid_type: int = 0,
        dx_const: float = 1000.0,
        dy_const: float = 1000.0,
        deglat: float = 15.0,
    ):
        self._grid_type = grid_type
        self._dx_const = dx_const
        self._dy_const = dy_const
        self._deglat = deglat
        self._halo = N_HALO_DEFAULT
        self._comm = communicator
        self._partitioner = self._comm.partitioner
        self._tile_partitioner = self._comm.tile.partitioner
        self._rank = self._comm.rank
        self.quantity_factory = quantity_factory
        self.quantity_factory.set_extra_dim_lengths(
            **{
                self.LON_OR_LAT_DIM: 2,
                self.TILE_DIM: 6,
                self.CARTESIAN_DIM: 3,
            }
        )
        self._grid_indexing = GridIndexing.from_sizer_and_communicator(
            self.quantity_factory.sizer, self._comm
        )
        self._grid_dims = [
            util.X_INTERFACE_DIM,
            util.Y_INTERFACE_DIM,
            self.LON_OR_LAT_DIM,
        ]
        self._grid_64 = self.quantity_factory.zeros(
            self._grid_dims,
            "radians",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        # This will carry the public version of the grid
        # for the selected floating point precision
        self._grid = None
        npx, npy, ndims = self._tile_partitioner.global_extent(self._grid_64)
        self._npx = npx
        self._npy = npy
        self._npz = self.quantity_factory.sizer.get_extent(util.Z_DIM)[0]
        self._agrid_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM, self.LON_OR_LAT_DIM],
            "radians",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        # This will carry the public version of the agrid
        # for the selected floating point precision
        self._agrid = None
        self._np = self._grid_64.np
        self._dx = None
        self._dy = None
        self._dx_agrid = None
        self._dy_agrid = None
        self._dx_center = None
        self._dy_center = None
        self._area = None
        self._area_c = None
        self._ks = None
        self._ak = None
        self._bk = None
        self._ptop = None
        self._ec1 = None
        self._ec2 = None
        self._ew1 = None
        self._ew2 = None
        self._es1 = None
        self._es2 = None
        self._ee1 = None
        self._ee2 = None
        self._l2c_v = None
        self._l2c_u = None
        self._cos_sg1 = None
        self._cos_sg2 = None
        self._cos_sg3 = None
        self._cos_sg4 = None
        self._cos_sg5 = None
        self._cos_sg6 = None
        self._cos_sg7 = None
        self._cos_sg8 = None
        self._cos_sg9 = None
        self._sin_sg1 = None
        self._sin_sg2 = None
        self._sin_sg3 = None
        self._sin_sg4 = None
        self._sin_sg5 = None
        self._sin_sg6 = None
        self._sin_sg7 = None
        self._sin_sg8 = None
        self._sin_sg9 = None
        self._cosa = None
        self._sina = None
        self._cosa_u = None
        self._cosa_v = None
        self._cosa_s = None
        self._sina_u = None
        self._sina_v = None
        self._rsin_u = None
        self._rsin_v = None
        self._rsina = None
        self._rsin2 = None
        self._del6_u = None
        self._del6_v = None
        self._divg_u = None
        self._divg_v = None
        self._vlon = None
        self._vlat = None
        self._z11 = None
        self._z12 = None
        self._z21 = None
        self._z22 = None
        self._a11 = None
        self._a12 = None
        self._a21 = None
        self._a22 = None
        self._edge_w = None
        self._edge_e = None
        self._edge_s = None
        self._edge_n = None
        self._edge_vect_w = None
        self._edge_vect_e = None
        self._edge_vect_s = None
        self._edge_vect_n = None
        self._edge_vect_w_2d = None
        self._edge_vect_e_2d = None
        self._da_min = None
        self._da_max = None
        self._da_min_c = None
        self._da_max_c = None

        # Data held for calculation only
        self._dx_64 = None
        self._dy_64 = None
        self._dxc_64 = None
        self._dyc_64 = None
        self._z11_64 = None
        self._z12_64 = None
        self._z21_64 = None
        self._z22_64 = None
        self._sina_u_64 = None
        self._sina_v_64 = None
        self._ec1_64 = None
        self._ec2_64 = None
        self._sin_sg5_64 = None
        self._vlon_64 = None
        self._vlat_64 = None

        # Initialize grids and configure internal numerics
        if grid_type == 4:
            self._compute_dxdy = self._compute_dxdy_cartesian
            self._compute_dxdy_agrid = self._compute_dxdy_agrid_cartesian
            self._compute_dxdy_center = self._compute_dxdy_center_cartesian
            self._compute_area = self._compute_area_cartesian
            self._compute_area_c = self._compute_area_c_cartesian
            self._calculate_center_vectors = self._calculate_center_vectors_cartesian
            self._calculate_vectors_west = self._calculate_vectors_west_cartesian
            self._calculate_vectors_south = self._calculate_vectors_south_cartesian
            self._init_cell_trigonometry = self._init_cell_trigonometry_cartesian
            self._calculate_latlon_momentum_correction = self._calculate_latlon_momentum_correction_cartesian
            self._calculate_xy_unit_vectors = self._calculate_xy_unit_vectors_cartesian
            self._calculate_unit_vectors_lonlat = self._calculate_unit_vectors_lonlat_cartesian
            self._init_cartesian()
        elif grid_type < 3:
            self._compute_dxdy = self._compute_dxdy_cube_sphere
            self._compute_dxdy_agrid = self._compute_dxdy_agrid_cube_sphere
            self._compute_dxdy_center = self._compute_dxdy_center_cube_sphere
            self._compute_area = self._compute_area_cube_sphere
            self._compute_area_c = self._compute_area_c_cube_sphere
            self._calculate_center_vectors = self._calculate_center_vectors_cube_sphere
            self._calculate_vectors_west = self._calculate_vectors_west_cube_sphere
            self._calculate_vectors_south = self._calculate_vectors_south_cube_sphere
            self._init_cell_trigonometry = self._init_cell_trigonometry_cube_sphere
            self._calculate_latlon_momentum_correction = self._calculate_latlon_momentum_correction_cube_sphere
            self._calculate_xy_unit_vectors = self._calculate_xy_unit_vectors_cube_sphere
            self._calculate_unit_vectors_lonlat = self._calculate_unit_vectors_lonlat_cube_sphere
            self._init_dgrid()
            self._init_agrid()
        else:
            raise NotImplementedError(f"Unsupported grid_type = {grid_type}")

    @classmethod
    def from_tile_sizing(
        cls,
        npx: int,
        npy: int,
        npz: int,
        communicator: util.CubedSphereCommunicator,
        backend: str,
        grid_type: int = 0,
        dx_const: float = 1000.0,
        dy_const: float = 1000.0,
        deglat: float = 15.0,
    ) -> "MetricTerms":
        sizer = util.SubtileGridSizer.from_tile_params(
            nx_tile=npx - 1,
            ny_tile=npy - 1,
            nz=npz,
            n_halo=N_HALO_DEFAULT,
            extra_dim_lengths={
                cls.LON_OR_LAT_DIM: 2,
                cls.TILE_DIM: 6,
                cls.CARTESIAN_DIM: 3,
            },
            layout=communicator.partitioner.tile.layout,
        )
        quantity_factory = util.QuantityFactory.from_backend(sizer, backend=backend)
        return cls(
            quantity_factory=quantity_factory,
            communicator=communicator,
            grid_type=grid_type,
            dx_const=dx_const,
            dy_const=dy_const,
            deglat=deglat,
        )

    @property
    def grid(self):
        if not self._grid:
            self._grid = quantity_cast_to_model_float(
                self.quantity_factory, self._grid_64
            )
        return self._grid

    @property
    def dgrid_lon_lat(self):
        """
        the longitudes and latitudes of the cell corners
        """
        return self.grid

    @property
    def gridvar(self):
        return self.grid

    @property
    def agrid(self):
        if not self._agrid:
            self._agrid = quantity_cast_to_model_float(
                self.quantity_factory, self._agrid_64
            )
        return self._agrid

    @property
    def agrid_lon_lat(self):
        """
        the longitudes and latitudes of the cell centers
        """
        return self.agrid

    @property
    def lon(self):
        return util.Quantity(
            data=self.grid.data[:, :, 0],
            dims=self.grid.dims[:2],
            origin=self.grid.origin[:2],
            extent=self.grid.extent[:2],
            units=self.grid.units,
            gt4py_backend=self.grid.gt4py_backend,
        )

    @property
    def lat(self) -> util.Quantity:
        return util.Quantity(
            data=self.grid.data[:, :, 1],
            dims=self.grid.dims[:2],
            origin=self.grid.origin[:2],
            extent=self.grid.extent[:2],
            units=self.grid.units,
            gt4py_backend=self.grid.gt4py_backend,
        )

    @property
    def lon_agrid(self) -> util.Quantity:
        return util.Quantity(
            data=self.agrid.data[:, :, 0],
            dims=self.agrid.dims[:2],
            origin=self.agrid.origin[:2],
            extent=self.agrid.extent[:2],
            units=self.agrid.units,
            gt4py_backend=self.agrid.gt4py_backend,
        )

    @property
    def lat_agrid(self) -> util.Quantity:
        return util.Quantity(
            data=self.agrid.data[:, :, 1],
            dims=self.agrid.dims[:2],
            origin=self.agrid.origin[:2],
            extent=self.agrid.extent[:2],
            units=self.agrid.units,
            gt4py_backend=self.agrid.gt4py_backend,
        )

    @property
    def dx(self) -> util.Quantity:
        """
        the distance between grid corners along the x-direction
        """
        if self._dx is None:
            self._dx, self._dy = self._compute_dxdy()
        return self._dx

    @property
    def dy(self) -> util.Quantity:
        """
        the distance between grid corners along the y-direction
        """
        if self._dy is None:
            self._dx, self._dy = self._compute_dxdy()
        return self._dy

    @property
    def dxa(self) -> util.Quantity:
        """
        the with of each grid cell along the x-direction
        """
        if self._dx_agrid is None:
            self._dx_agrid, self._dy_agrid = self._compute_dxdy_agrid()
        return self._dx_agrid

    @property
    def dya(self) -> util.Quantity:
        """
        the with of each grid cell along the y-direction
        """
        if self._dy_agrid is None:
            self._dx_agrid, self._dy_agrid = self._compute_dxdy_agrid()
        return self._dy_agrid

    @property
    def dxc(self) -> util.Quantity:
        """
        the distance between cell centers along the x-direction
        """
        if self._dx_center is None:
            self._dx_center, self._dy_center = self._compute_dxdy_center()
        return self._dx_center

    @property
    def dyc(self) -> util.Quantity:
        """
        the distance between cell centers along the y-direction
        """
        if self._dy_center is None:
            self._dx_center, self._dy_center = self._compute_dxdy_center()
        return self._dy_center

    @property
    def ks(self) -> util.Quantity:
        """
        number of levels where the vertical coordinate is purely pressure-based
        """
        if self._ks is None:
            (
                self._ks,
                self._ptop,
                self._ak,
                self._bk,
            ) = self._set_hybrid_pressure_coefficients()
        return self._ks

    @property
    def ak(self) -> util.Quantity:
        """
        the ak coefficient used to calculate the pressure at a given k-level:
        pk = ak + (bk * ps)
        """
        if self._ak is None:
            (
                self._ks,
                self._ptop,
                self._ak,
                self._bk,
            ) = self._set_hybrid_pressure_coefficients()
        return self._ak

    @property
    def bk(self) -> util.Quantity:
        """
        the bk coefficient used to calculate the pressure at a given k-level:
        pk = ak + (bk * ps)
        """
        if self._bk is None:
            (
                self._ks,
                self._ptop,
                self._ak,
                self._bk,
            ) = self._set_hybrid_pressure_coefficients()
        return self._bk

    @property
    def ptop(self) -> util.Quantity:
        """
        the pressure of the top of atmosphere level
        """
        if self._ptop is None:
            (
                self._ks,
                self._ptop,
                self._ak,
                self._bk,
            ) = self._set_hybrid_pressure_coefficients()
        return self._ptop

    @property
    def ec1(self) -> util.Quantity:
        """
        cartesian components of the local unit vetcor
        in the x-direction at the cell centers
        3d array whose last dimension is length 3 and indicates cartesian x/y/z value
        """
        if self._ec1 is None:
            self._ec1, self._ec2 = self._calculate_center_vectors()
        return self._ec1

    @property
    def ec2(self) -> util.Quantity:
        """
        cartesian components of the local unit vetcor
        in the y-direation at the cell centers
        3d array whose last dimension is length 3 and indicates cartesian x/y/z value
        """
        if self._ec2 is None:
            self._ec1, self._ec2 = self._calculate_center_vectors()
        return self._ec2

    @property
    def ew1(self) -> util.Quantity:
        """
        cartesian components of the local unit vetcor
        in the x-direation at the left/right cell edges
        3d array whose last dimension is length 3 and indicates cartesian x/y/z value
        """
        if self._ew1 is None:
            self._ew1, self._ew2 = self._calculate_vectors_west()
        return self._ew1

    @property
    def ew2(self) -> util.Quantity:
        """
        cartesian components of the local unit vetcor
        in the y-direation at the left/right cell edges
        3d array whose last dimension is length 3 and indicates cartesian x/y/z value
        """
        if self._ew2 is None:
            self._ew1, self._ew2 = self._calculate_vectors_west()
        return self._ew2

    @property
    def cos_sg1(self) -> util.Quantity:
        """
        Cosine of the angle at point 1 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        """
        if self._cos_sg1 is None:
            self._init_cell_trigonometry()
        return self._cos_sg1

    @property
    def cos_sg2(self) -> util.Quantity:
        """
        Cosine of the angle at point 2 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        """
        if self._cos_sg2 is None:
            self._init_cell_trigonometry()
        return self._cos_sg2

    @property
    def cos_sg3(self) -> util.Quantity:
        """
        Cosine of the angle at point 3 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        """
        if self._cos_sg3 is None:
            self._init_cell_trigonometry()
        return self._cos_sg3

    @property
    def cos_sg4(self) -> util.Quantity:
        """
        Cosine of the angle at point 4 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        """
        if self._cos_sg4 is None:
            self._init_cell_trigonometry()
        return self._cos_sg4

    @property
    def cos_sg5(self) -> util.Quantity:
        """
        Cosine of the angle at point 5 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        The inner product of ec1 and ec2 for point 5
        """
        if self._cos_sg5 is None:
            self._init_cell_trigonometry()
        return self._cos_sg5

    @property
    def cos_sg6(self) -> util.Quantity:
        """
        Cosine of the angle at point 6 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        """
        if self._cos_sg6 is None:
            self._init_cell_trigonometry()
        return self._cos_sg6

    @property
    def cos_sg7(self) -> util.Quantity:
        """
        Cosine of the angle at point 7 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        """
        if self._cos_sg7 is None:
            self._init_cell_trigonometry()
        return self._cos_sg7

    @property
    def cos_sg8(self) -> util.Quantity:
        """
        Cosine of the angle at point 8 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        """
        if self._cos_sg8 is None:
            self._init_cell_trigonometry()
        return self._cos_sg8

    @property
    def cos_sg9(self) -> util.Quantity:
        """
        Cosine of the angle at point 9 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        """
        if self._cos_sg9 is None:
            self._init_cell_trigonometry()
        return self._cos_sg9

    @property
    def sin_sg1(self) -> util.Quantity:
        """
        Sine of the angle at point 1 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        """
        if self._sin_sg1 is None:
            self._init_cell_trigonometry()
        return self._sin_sg1

    @property
    def sin_sg2(self) -> util.Quantity:
        """
        Sine of the angle at point 2 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        """
        if self._sin_sg2 is None:
            self._init_cell_trigonometry()
        return self._sin_sg2

    @property
    def sin_sg3(self) -> util.Quantity:
        """
        Sine of the angle at point 3 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        """
        if self._sin_sg3 is None:
            self._init_cell_trigonometry()
        return self._sin_sg3

    @property
    def sin_sg4(self) -> util.Quantity:
        """
        Sine of the angle at point 4 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        """
        if self._sin_sg4 is None:
            self._init_cell_trigonometry()
        return self._sin_sg4

    @property
    def sin_sg5(self) -> util.Quantity:
        """
        Sine of the angle at point 5 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        For the center point this is one minus the inner product of ec1 and ec2 squared
        """
        if self._sin_sg5 is None:
            self._init_cell_trigonometry()
        return self._sin_sg5

    @property
    def sin_sg6(self) -> util.Quantity:
        """
        Sine of the angle at point 6 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        """
        if self._sin_sg6 is None:
            self._init_cell_trigonometry()
        return self._sin_sg6

    @property
    def sin_sg7(self) -> util.Quantity:
        """
        Sine of the angle at point 7 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        """
        if self._sin_sg7 is None:
            self._init_cell_trigonometry()
        return self._sin_sg7

    @property
    def sin_sg8(self) -> util.Quantity:
        """
        Sine of the angle at point 8 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        """
        if self._sin_sg8 is None:
            self._init_cell_trigonometry()
        return self._sin_sg8

    @property
    def sin_sg9(self) -> util.Quantity:
        """
        Sine of the angle at point 9 of the 'supergrid' within each grid cell:
        9---4---8
        |       |
        1   5   3
        |       |
        6---2---7
        """
        if self._sin_sg9 is None:
            self._init_cell_trigonometry()
        return self._sin_sg9

    @property
    def cosa(self) -> util.Quantity:
        """
        cosine of angle between coordinate lines at the cell corners
        averaged to ensure consistent answers
        """
        if self._cosa is None:
            self._init_cell_trigonometry()
        return self._cosa

    @property
    def sina(self) -> util.Quantity:
        """
        as cosa but sine
        """
        if self._sina is None:
            self._init_cell_trigonometry()
        return self._sina

    @property
    def cosa_u(self) -> util.Quantity:
        """
        as cosa but defined at the left and right cell edges
        """
        if self._cosa_u is None:
            self._init_cell_trigonometry()
        return self._cosa_u

    @property
    def cosa_v(self) -> util.Quantity:
        """
        as cosa but defined at the top and bottom cell edges
        """
        if self._cosa_v is None:
            self._init_cell_trigonometry()
        return self._cosa_v

    @property
    def cosa_s(self) -> util.Quantity:
        """
        as cosa but defined at cell centers
        """
        if self._cosa_s is None:
            self._init_cell_trigonometry()
        return self._cosa_s

    @property
    def sina_u(self) -> util.Quantity:
        """
        as cosa_u but with sine
        """
        if self._sina_u is None:
            self._init_cell_trigonometry()
        return self._sina_u

    @property
    def sina_v(self) -> util.Quantity:
        """
        as cosa_v but with sine
        """
        if self._sina_v is None:
            self._init_cell_trigonometry()
        return self._sina_v

    @property
    def rsin_u(self) -> util.Quantity:
        """
        1/sina_u**2,
        defined as the inverse-squrared as it is only used as such
        """
        if self._rsin_u is None:
            self._init_cell_trigonometry()
        return self._rsin_u

    @property
    def rsin_v(self) -> util.Quantity:
        """
        1/sina_v**2,
        defined as the inverse-squrared as it is only used as such
        """
        if self._rsin_v is None:
            self._init_cell_trigonometry()
        return self._rsin_v

    @property
    def rsina(self) -> util.Quantity:
        """
        1/sina**2,
        defined as the inverse-squrared as it is only used as such
        """
        if self._rsina is None:
            self._init_cell_trigonometry()
        return self._rsina

    @property
    def rsin2(self) -> util.Quantity:
        """
        1/sin_sg5**2,
        defined as the inverse-squrared as it is only used as such
        """
        if self._rsin2 is None:
            self._init_cell_trigonometry()
        return self._rsin2

    @property
    def l2c_v(self) -> util.Quantity:
        """
        angular momentum correction for converting v-winds
        from lat/lon to cartesian coordinates
        """
        if self._l2c_v is None:
            self._l2c_v, self._l2c_u = self._calculate_latlon_momentum_correction()
        return self._l2c_v

    @property
    def l2c_u(self) -> util.Quantity:
        """
        angular momentum correction for converting u-winds
        from lat/lon to cartesian coordinates
        """
        if self._l2c_u is None:
            self._l2c_v, self._l2c_u = self._calculate_latlon_momentum_correction()
        return self._l2c_u

    @property
    def es1(self) -> util.Quantity:
        """
        cartesian components of the local unit vetcor
        in the x-direation at the top/bottom cell edges,
        3d array whose last dimension is length 3 and indicates cartesian x/y/z value
        """
        if self._es1 is None:
            self._es1, self._es2 = self._calculate_vectors_south()
        return self._es1

    @property
    def es2(self) -> util.Quantity:
        """
        cartesian components of the local unit vetcor
        in the y-direation at the top/bottom cell edges,
        3d array whose last dimension is length 3 and indicates cartesian x/y/z value
        """
        if self._es2 is None:
            self._es1, self._es2 = self._calculate_vectors_south()
        return self._es2

    @property
    def ee1(self) -> util.Quantity:
        """
        cartesian components of the local unit vetcor
        in the x-direation at the cell corners,
        3d array whose last dimension is length 3 and indicates cartesian x/y/z value
        """
        if self._ee1 is None:
            self._ee1, self._ee2 = self._calculate_xy_unit_vectors()
        return self._ee1

    @property
    def ee2(self) -> util.Quantity:
        """
        cartesian components of the local unit vetcor
        in the y-direation at the cell corners,
        3d array whose last dimension is length 3 and indicates cartesian x/y/z value
        """
        if self._ee2 is None:
            self._ee1, self._ee2 = self._calculate_xy_unit_vectors()
        return self._ee2

    @property
    def divg_u(self) -> util.Quantity:
        """
        sina_v * dyc/dx
        """
        if self._divg_u is None:
            (
                self._del6_u,
                self._del6_v,
                self._divg_u,
                self._divg_v,
            ) = self._calculate_divg_del6()
        return self._divg_u

    @property
    def divg_v(self) -> util.Quantity:
        """
        sina_u * dxc/dy
        """
        if self._divg_v is None:
            (
                self._del6_u,
                self._del6_v,
                self._divg_u,
                self._divg_v,
            ) = self._calculate_divg_del6()
        return self._divg_v

    @property
    def del6_u(self) -> util.Quantity:
        """
        sina_v * dx/dyc
        """
        if self._del6_u is None:
            (
                self._del6_u,
                self._del6_v,
                self._divg_u,
                self._divg_v,
            ) = self._calculate_divg_del6()
        return self._del6_u

    @property
    def del6_v(self) -> util.Quantity:
        """
        sina_u * dy/dxc
        """
        if self._del6_v is None:
            (
                self._del6_u,
                self._del6_v,
                self._divg_u,
                self._divg_v,
            ) = self._calculate_divg_del6()
        return self._del6_v

    @property
    def vlon(self) -> util.Quantity:
        """
        unit vector in eastward longitude direction,
        3d array whose last dimension is length 3 and indicates x/y/z value
        """
        if self._vlon is None:
            self._vlon, self._vlat = self._calculate_unit_vectors_lonlat()
        return self._vlon

    @property
    def vlat(self) -> util.Quantity:
        """
        unit vector in northward latitude direction,
        3d array whose last dimension is length 3 and indicates x/y/z value
        """
        if self._vlat is None:
            self._vlon, self._vlat = self._calculate_unit_vectors_lonlat()
        return self._vlat

    @property
    def z11(self) -> util.Quantity:
        """
        vector product of horizontal component of the cell-center vector
        with the unit longitude vector
        """
        if self._z11 is None:
            self._z11, self._z12, self._z21, self._z22 = self._calculate_grid_z()
        return self._z11

    @property
    def z12(self) -> util.Quantity:
        """
        vector product of horizontal component of the cell-center vector
        with the unit latitude vector
        """
        if self._z12 is None:
            self._z11, self._z12, self._z21, self._z22 = self._calculate_grid_z()
        return self._z12

    @property
    def z21(self) -> util.Quantity:
        """
        vector product of vertical component of the cell-center vector
        with the unit longitude vector
        """
        if self._z21 is None:
            self._z11, self._z12, self._z21, self._z22 = self._calculate_grid_z()
        return self._z21

    @property
    def z22(self) -> util.Quantity:
        """
        vector product of vertical component of the cell-center vector
        with the unit latitude vector
        """
        if self._z22 is None:
            self._z11, self._z12, self._z21, self._z22 = self._calculate_grid_z()
        return self._z22

    @property
    def a11(self) -> util.Quantity:
        """
        0.5*z22/sin_sg5
        """
        if self._a11 is None:
            self._a11, self._a12, self._a21, self._a22 = self._calculate_grid_a()
        return self._a11

    @property
    def a12(self) -> util.Quantity:
        """
        0.5*z21/sin_sg5
        """
        if self._a12 is None:
            self._a11, self._a12, self._a21, self._a22 = self._calculate_grid_a()
        return self._a12

    @property
    def a21(self) -> util.Quantity:
        """
        0.5*z12/sin_sg5
        """
        if self._a21 is None:
            self._a11, self._a12, self._a21, self._a22 = self._calculate_grid_a()
        return self._a21

    @property
    def a22(self) -> util.Quantity:
        """
        0.5*z11/sin_sg5
        """
        if self._a22 is None:
            self._a11, self._a12, self._a21, self._a22 = self._calculate_grid_a()
        return self._a22

    @property
    def edge_w(self) -> util.Quantity:
        """
        factor to interpolate scalars from a to c grid at the western grid edge
        """
        if self._edge_w is None:
            (
                self._edge_w,
                self._edge_e,
                self._edge_s,
                self._edge_n,
            ) = self._calculate_edge_factors()
        return self._edge_w

    @property
    def edge_e(self) -> util.Quantity:
        """
        factor to interpolate scalars from a to c grid at the eastern grid edge
        """
        if self._edge_e is None:
            (
                self._edge_w,
                self._edge_e,
                self._edge_s,
                self._edge_n,
            ) = self._calculate_edge_factors()
        return self._edge_e

    @property
    def edge_s(self) -> util.Quantity:
        """
        factor to interpolate scalars from a to c grid at the southern grid edge
        """
        if self._edge_s is None:
            (
                self._edge_w,
                self._edge_e,
                self._edge_s,
                self._edge_n,
            ) = self._calculate_edge_factors()
        return self._edge_s

    @property
    def edge_n(self) -> util.Quantity:
        """
        factor to interpolate scalars from a to c grid at the northern grid edge
        """
        if self._edge_n is None:
            (
                self._edge_w,
                self._edge_e,
                self._edge_s,
                self._edge_n,
            ) = self._calculate_edge_factors()
        return self._edge_n

    @property
    def edge_vect_w_1d(self) -> util.Quantity:
        """
        factor to interpolate vectors from a to c grid at the western grid edge
        """
        if self._edge_vect_w is None:
            (
                self._edge_vect_w,
                self._edge_vect_e,
                self._edge_vect_s,
                self._edge_vect_n,
            ) = self._calculate_edge_a2c_vect_factors()
        return self._edge_vect_w

    @property
    def edge_vect_w(self) -> util.Quantity:
        """
        factor to interpolate vectors from a to c grid at the western grid edge
        repeated in x and y to be used in stencils
        """
        if self._edge_vect_w_2d is None:
            (
                self._edge_vect_e_2d,
                self._edge_vect_w_2d,
            ) = self._calculate_2d_edge_a2c_vect_factors()
        return self._edge_vect_w_2d

    @property
    def edge_vect_e_1d(self) -> util.Quantity:
        """
        factor to interpolate vectors from a to c grid at the eastern grid edge
        """
        if self._edge_vect_e is None:
            (
                self._edge_vect_w,
                self._edge_vect_e,
                self._edge_vect_s,
                self._edge_vect_n,
            ) = self._calculate_edge_a2c_vect_factors()
        return self._edge_vect_e

    @property
    def edge_vect_e(self) -> util.Quantity:
        """
        factor to interpolate vectors from a to c grid at the eastern grid edge
        repeated in x and y to be used in stencils
        """
        if self._edge_vect_e_2d is None:
            (
                self._edge_vect_e_2d,
                self._edge_vect_w_2d,
            ) = self._calculate_2d_edge_a2c_vect_factors()
        return self._edge_vect_e_2d

    @property
    def edge_vect_s(self) -> util.Quantity:
        """
        factor to interpolate vectors from a to c grid at the southern grid edge
        """
        if self._edge_vect_s is None:
            (
                self._edge_vect_w,
                self._edge_vect_e,
                self._edge_vect_s,
                self._edge_vect_n,
            ) = self._calculate_edge_a2c_vect_factors()
        return self._edge_vect_s

    @property
    def edge_vect_n(self) -> util.Quantity:
        """
        factor to interpolate vectors from a to c grid at the northern grid edge
        """
        if self._edge_vect_n is None:
            (
                self._edge_vect_w,
                self._edge_vect_e,
                self._edge_vect_s,
                self._edge_vect_n,
            ) = self._calculate_edge_a2c_vect_factors()
        return self._edge_vect_n

    @property
    def da_min(self) -> float:
        """
        the minimum agrid cell area across all ranks,
        if mpi is not present and the communicator is a DummyComm this will be
        the minimum on the local rank
        """
        if self._da_min is None:
            self._reduce_global_area_minmaxes()
        return self._da_min

    @property
    def da_max(self) -> float:
        """
        the maximum agrid cell area across all ranks,
        if mpi is not present and the communicator is a DummyComm this will be
        the maximum on the local rank
        """
        if self._da_max is None:
            self._reduce_global_area_minmaxes()
        return self._da_max

    @property
    def da_min_c(self) -> float:
        """
        the minimum cgrid cell area across all ranks,
        if mpi is not present and the communicator is a DummyComm this will be
        the minimum on the local rank
        """
        if self._da_min_c is None:
            self._reduce_global_area_minmaxes()
        return self._da_min_c

    @property
    def da_max_c(self) -> float:
        """
        the maximum cgrid cell area across all ranks,
        if mpi is not present and the communicator is a DummyComm this will be
        the maximum on the local rank
        """
        if self._da_max_c is None:
            self._reduce_global_area_minmaxes()
        return self._da_max_c

    @property
    def area(self) -> util.Quantity:
        """
        the area of each a-grid cell
        """
        if self._area is None:
            self._area = self._compute_area()
        return self._area

    @property
    def area_c(self) -> util.Quantity:
        """
        the area of each c-grid cell
        """
        if self._area_c is None:
            self._area_c = self._compute_area_c()
        return self._area_c

    @cached_property
    def _dgrid_xyz_64(self) -> util.Quantity:
        """
        cartesian coordinates of each dgrid cell center
        """
        return lon_lat_to_xyz(
            self._grid_64.data[:, :, 0], self._grid_64.data[:, :, 1], self._np
        )

    @cached_property
    def _agrid_xyz_64(self) -> util.Quantity:
        """
        cartesian coordinates of each agrid cell center
        """
        return lon_lat_to_xyz(
            self._agrid_64.data[:-1, :-1, 0],
            self._agrid_64.data[:-1, :-1, 1],
            self._np,
        )

    @cached_property
    def rarea(self) -> util.Quantity:
        """
        1/cell area
        """
        return util.Quantity(
            data=1.0 / self.area.data,
            dims=self.area.dims,
            origin=self.area.origin,
            extent=self.area.extent,
            units="m^-2",
            gt4py_backend=self.area.gt4py_backend,
        )

    @cached_property
    def rarea_c(self) -> util.Quantity:
        """
        1/cgrid cell area
        """
        return util.Quantity(
            data=1.0 / self.area_c.data,
            dims=self.area_c.dims,
            origin=self.area_c.origin,
            extent=self.area_c.extent,
            units="m^-2",
            gt4py_backend=self.area_c.gt4py_backend,
        )

    @cached_property
    @ignore_zero_division
    def rdx(self) -> util.Quantity:
        """
        1/dx
        """
        return util.Quantity(
            data=1.0 / self.dx.data,
            dims=self.dx.dims,
            origin=self.dx.origin,
            extent=self.dx.extent,
            units="m^-1",
            gt4py_backend=self.dx.gt4py_backend,
        )

    @cached_property
    @ignore_zero_division
    def rdy(self) -> util.Quantity:
        """
        1/dy
        """
        return util.Quantity(
            data=1.0 / self.dy.data,
            dims=self.dy.dims,
            origin=self.dy.origin,
            extent=self.dy.extent,
            units="m^-1",
            gt4py_backend=self.dy.gt4py_backend,
        )

    @cached_property
    @ignore_zero_division
    def rdxa(self) -> util.Quantity:
        """
        1/dxa
        """
        return util.Quantity(
            data=1.0 / self.dxa.data,
            dims=self.dxa.dims,
            origin=self.dxa.origin,
            extent=self.dxa.extent,
            units="m^-1",
            gt4py_backend=self.dxa.gt4py_backend,
        )

    @cached_property
    @ignore_zero_division
    def rdya(self) -> util.Quantity:
        """
        1/dya
        """
        return util.Quantity(
            data=1.0 / self.dya.data,
            dims=self.dya.dims,
            origin=self.dya.origin,
            extent=self.dya.extent,
            units="m^-1",
            gt4py_backend=self.dya.gt4py_backend,
        )

    @cached_property
    @ignore_zero_division
    def rdxc(self) -> util.Quantity:
        """
        1/dxc
        """
        return util.Quantity(
            data=1.0 / self.dxc.data,
            dims=self.dxc.dims,
            origin=self.dxc.origin,
            extent=self.dxc.extent,
            units="m^-1",
            gt4py_backend=self.dxc.gt4py_backend,
        )

    @cached_property
    @ignore_zero_division
    def rdyc(self) -> util.Quantity:
        """
        1/dyc
        """
        return util.Quantity(
            data=1.0 / self.dyc.data,
            dims=self.dyc.dims,
            origin=self.dyc.origin,
            extent=self.dyc.extent,
            units="m^-1",
            gt4py_backend=self.dyc.gt4py_backend,
        )

    def _init_cartesian(self):

        domain_rad = PI / 16.0
        lat_rad = self._deglat * PI / 180.0
        lon_rad = 0.0

        self._grid_64.data[:, :, :] = self._np.nan
        slice_x, slice_y = self._tile_partitioner.subtile_slice(
            self._rank, self._grid_64.dims, (self._npx, self._npy)
        )

        isd = slice_x.start - self._halo
        ied = slice_x.stop + self._halo
        isg = max(isd, 0)
        ieg = min(ied, self._npx)
        isl = isg - isd
        iel = isl + ieg - isg

        jsd = slice_y.start - self._halo
        jed = slice_y.stop + self._halo
        jsg = max(jsd, 0)
        jeg = min(jed, self._npy)
        jsl = jsg - jsd
        jel = jsl + jeg - jsg

        lon_frac = np.array(range(isg, ieg)) / (self._npx - 1) - 0.5
        lon_frac = lon_frac[:, np.newaxis]
        lat_frac = np.array(range(jsg, jeg)) / (self._npy - 1) - 0.5
        lat_frac = lat_frac[np.newaxis, :]

        self._grid_64.data[isl:iel, jsl:jel, 0] = lon_rad + lon_frac * domain_rad
        self._grid_64.data[isl:iel, jsl:jel, 1] = lat_rad + lat_frac * domain_rad

        self._agrid_64.data[:, :, 0] = lon_rad
        self._agrid_64.data[:, :, 1] = lat_rad

    def _init_dgrid(self):
        grid_mirror_ew = self.quantity_factory.zeros(
            self._grid_dims,
            "radians",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        grid_mirror_ns = self.quantity_factory.zeros(
            self._grid_dims,
            "radians",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        grid_mirror_diag = self.quantity_factory.zeros(
            self._grid_dims,
            "radians",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )

        local_west_edge = self._tile_partitioner.on_tile_left(self._rank)
        local_east_edge = self._tile_partitioner.on_tile_right(self._rank)
        local_south_edge = self._tile_partitioner.on_tile_bottom(self._rank)
        local_north_edge = self._tile_partitioner.on_tile_top(self._rank)
        # information on position of subtile in full tile
        slice_x, slice_y = self._tile_partitioner.subtile_slice(
            self._rank, self._grid_64.dims, (self._npx, self._npy), overlap=True
        )
        section_global_is = self._halo + slice_x.start
        section_global_js = self._halo + slice_y.start
        subtile_width_x = slice_x.stop - slice_x.start - 1
        subtile_width_y = slice_y.stop - slice_y.start - 1

        # compute gnomonic grid for this rank
        local_gnomonic_ed(
            self._grid_64.view[:, :, 0],
            self._grid_64.view[:, :, 1],
            npx=self._npx,
            west_edge=local_west_edge,
            east_edge=local_east_edge,
            south_edge=local_south_edge,
            north_edge=local_north_edge,
            global_is=section_global_is,
            global_js=section_global_js,
            np=self._np,
            rank=self._rank,
        )

        # Next compute gnomonic for the mirrored ranks that'll be averaged
        j_subtile_index, i_subtile_index = self._tile_partitioner.subtile_index(
            self._rank
        )
        # compute the global index starting points for the mirrored ranks
        ew_global_is = (
            self._halo
            + (self._tile_partitioner.layout[0] - i_subtile_index - 1) * subtile_width_x
        )
        ns_global_js = (
            self._halo
            + (self._tile_partitioner.layout[1] - j_subtile_index - 1) * subtile_width_y
        )

        # compute mirror in the east-west direction
        west_edge = True if local_east_edge else False
        east_edge = True if local_west_edge else False
        local_gnomonic_ed(
            grid_mirror_ew.view[:, :, 0],
            grid_mirror_ew.view[:, :, 1],
            npx=self._npx,
            west_edge=west_edge,
            east_edge=east_edge,
            south_edge=local_south_edge,
            north_edge=local_north_edge,
            global_is=ew_global_is,
            global_js=section_global_js,
            np=self._np,
            rank=self._rank,
        )

        # compute mirror in the north-south direction
        south_edge = True if local_north_edge else False
        north_edge = True if local_south_edge else False
        local_gnomonic_ed(
            grid_mirror_ns.view[:, :, 0],
            grid_mirror_ns.view[:, :, 1],
            npx=self._npx,
            west_edge=local_west_edge,
            east_edge=local_east_edge,
            south_edge=south_edge,
            north_edge=north_edge,
            global_is=section_global_is,
            global_js=ns_global_js,
            np=self._np,
            rank=self._rank,
        )

        local_gnomonic_ed(
            grid_mirror_diag.view[:, :, 0],
            grid_mirror_diag.view[:, :, 1],
            npx=self._npx,
            west_edge=west_edge,
            east_edge=east_edge,
            south_edge=south_edge,
            north_edge=north_edge,
            global_is=ew_global_is,
            global_js=ns_global_js,
            np=self._np,
            rank=self._rank,
        )

        # Average the mirrored gnomonic grids
        tile_index = self._partitioner.tile_index(self._rank)
        mirror_data = {
            "local": self._grid_64.data,
            "east-west": grid_mirror_ew.data,
            "north-south": grid_mirror_ns.data,
            "diagonal": grid_mirror_diag.data,
        }
        mirror_grid(
            mirror_data=mirror_data,
            tile_index=tile_index,
            npx=self._npx,
            npy=self._npy,
            x_subtile_width=subtile_width_x + 1,
            y_subtile_width=subtile_width_y + 1,
            global_is=section_global_is,
            global_js=section_global_js,
            ng=self._halo,
            np=self._grid_64.np,
            right_hand_grid=self.RIGHT_HAND_GRID,
        )

        # Shift the corner away from Japan
        # This will result in the corner close to east coast of China
        # TODO if not config.do_schmidt and config.shift_fac > 1.0e-4
        shift_fac = 18
        self._grid_64.view[:, :, 0] -= PI / shift_fac
        tile0_lon = self._grid_64.data[:, :, 0]
        tile0_lon[tile0_lon < 0] += 2 * PI
        self._grid_64.data[self._np.abs(self._grid_64.data[:]) < 1e-10] = 0.0

        self._comm.halo_update(self._grid_64, n_points=self._halo)

        fill_corners_2d(
            self._grid_64.data, self._grid_indexing, gridtype="B", direction="x"
        )

    def _init_agrid(self):
        # Set up lat-lon a-grid, calculate side lengths on a-grid
        lon_agrid, lat_agrid = lon_lat_corner_to_cell_center(
            self._grid_64.data[:, :, 0], self._grid_64.data[:, :, 1], self._np
        )
        self._agrid_64.data[:-1, :-1, 0], self._agrid_64.data[:-1, :-1, 1] = (
            lon_agrid,
            lat_agrid,
        )
        self._comm.halo_update(self._agrid_64, n_points=self._halo)
        fill_corners_2d(
            self._agrid_64.data[:, :, 0][:, :, None],
            self._grid_indexing,
            gridtype="A",
            direction="x",
        )
        fill_corners_2d(
            self._agrid_64.data[:, :, 1][:, :, None],
            self._grid_indexing,
            gridtype="A",
            direction="y",
        )

    def _compute_dxdy_cube_sphere(self):
        dx_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "m",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )

        dx_64.view[:, :] = great_circle_distance_along_axis(
            self._grid_64.view[:, :, 0],
            self._grid_64.view[:, :, 1],
            RADIUS,
            self._np,
            axis=0,
        )
        dy_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "m",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        dy_64.view[:, :] = great_circle_distance_along_axis(
            self._grid_64.view[:, :, 0],
            self._grid_64.view[:, :, 1],
            RADIUS,
            self._np,
            axis=1,
        )
        self._comm.vector_halo_update(dx_64, dy_64, n_points=self._halo)

        # at this point the Fortran code copies in the west and east edges from
        # the halo for dy and performs a halo update,
        # to ensure dx and dy mirror across the boundary.
        # Not doing it here at the moment.
        dx_64.data[dx_64.data < 0] *= -1
        dy_64.data[dy_64.data < 0] *= -1
        fill_corners_dgrid(
            dx_64.data[:, :, None],
            dy_64.data[:, :, None],
            self._grid_indexing,
            vector=False,
        )

        dx = quantity_cast_to_model_float(self.quantity_factory, dx_64)
        self._dx_64 = dx_64
        dy = quantity_cast_to_model_float(self.quantity_factory, dy_64)
        self._dy_64 = dy_64

        return dx, dy

    def _compute_dxdy_cartesian(self):
        dx_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "m",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        dx_64.data[:, :] = self._dx_const

        dy_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "m",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        dy_64.data[:, :] = self._dy_const

        dx = quantity_cast_to_model_float(self.quantity_factory, dx_64)
        self._dx_64 = dx_64
        dy = quantity_cast_to_model_float(self.quantity_factory, dy_64)
        self._dy_64 = dy_64

        return dx, dy

    def _compute_dxdy_agrid_cube_sphere(self):
        dx_agrid_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "m",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        dy_agrid_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "m",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        lon, lat = self._grid_64.data[:, :, 0], self._grid_64.data[:, :, 1]
        lon_y_center, lat_y_center = lon_lat_midpoint(
            lon[:, :-1], lon[:, 1:], lat[:, :-1], lat[:, 1:], self._np
        )
        dx_agrid_tmp = great_circle_distance_along_axis(
            lon_y_center, lat_y_center, RADIUS, self._np, axis=0
        )
        lon_x_center, lat_x_center = lon_lat_midpoint(
            lon[:-1, :], lon[1:, :], lat[:-1, :], lat[1:, :], self._np
        )
        dy_agrid_tmp = great_circle_distance_along_axis(
            lon_x_center, lat_x_center, RADIUS, self._np, axis=1
        )
        fill_corners_agrid(
            dx_agrid_tmp[:, :, None],
            dy_agrid_tmp[:, :, None],
            self._grid_indexing,
            vector=False,
        )

        dx_agrid_64.data[:-1, :-1] = dx_agrid_tmp
        dy_agrid_64.data[:-1, :-1] = dy_agrid_tmp
        self._comm.vector_halo_update(dx_agrid_64, dy_agrid_64, n_points=self._halo)

        # at this point the Fortran code copies in the west and east edges from
        # the halo for dy and performs a halo update,
        # to ensure dx and dy mirror across the boundary.
        # Not doing it here at the moment.
        dx_agrid_64.data[dx_agrid_64.data < 0] *= -1
        dy_agrid_64.data[dy_agrid_64.data < 0] *= -1

        dx_agrid = quantity_cast_to_model_float(self.quantity_factory, dx_agrid_64)
        dy_agrid = quantity_cast_to_model_float(self.quantity_factory, dy_agrid_64)

        return dx_agrid, dy_agrid

    def _compute_dxdy_agrid_cartesian(self):
        dx_agrid_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "m",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        dx_agrid_64.data[:, :] = self._dx_const

        dy_agrid_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "m",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        dy_agrid_64.data[:, :] = self._dy_const

        dx_agrid = quantity_cast_to_model_float(self.quantity_factory, dx_agrid_64)
        dy_agrid = quantity_cast_to_model_float(self.quantity_factory, dy_agrid_64)

        return dx_agrid, dy_agrid

    def _compute_dxdy_center_cube_sphere(self):
        dx_center_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "m",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        dy_center_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "m",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )

        lon_agrid, lat_agrid = (
            self._agrid_64.data[:-1, :-1, 0],
            self._agrid_64.data[:-1, :-1, 1],
        )
        dx_center_tmp = great_circle_distance_along_axis(
            lon_agrid, lat_agrid, RADIUS, self._np, axis=0
        )
        dy_center_tmp = great_circle_distance_along_axis(
            lon_agrid, lat_agrid, RADIUS, self._np, axis=1
        )
        # copying the second-to-last values to the last values is what the Fortran
        # code does, but is this correct/valid?
        # Maybe we want to change this to use halo updates?
        dx_center_64.data[1:-1, :-1] = dx_center_tmp
        dx_center_64.data[0, :-1] = dx_center_tmp[0, :]
        dx_center_64.data[-1, :-1] = dx_center_tmp[-1, :]

        dy_center_64.data[:-1, 1:-1] = dy_center_tmp
        dy_center_64.data[:-1, 0] = dy_center_tmp[:, 0]
        dy_center_64.data[:-1, -1] = dy_center_tmp[:, -1]

        set_tile_border_dxc(
            self._dgrid_xyz_64[3:-3, 3:-3, :],
            self._agrid_xyz_64[3:-3, 3:-3, :],
            RADIUS,
            dx_center_64.data[3:-3, 3:-4],
            self._tile_partitioner,
            self._rank,
            self._np,
        )
        set_tile_border_dyc(
            self._dgrid_xyz_64[3:-3, 3:-3, :],
            self._agrid_xyz_64[3:-3, 3:-3, :],
            RADIUS,
            dy_center_64.data[3:-4, 3:-3],
            self._tile_partitioner,
            self._rank,
            self._np,
        )
        self._comm.vector_halo_update(dx_center_64, dy_center_64, n_points=self._halo)

        # TODO: Add support for unsigned vector halo updates
        # instead of handling ad-hoc here
        dx_center_64.data[dx_center_64.data < 0] *= -1
        dy_center_64.data[dy_center_64.data < 0] *= -1

        # TODO: fix issue with interface dimensions causing validation errors
        fill_corners_cgrid(
            dx_center_64.data[:, :, None],
            dy_center_64.data[:, :, None],
            self._grid_indexing,
            vector=False,
        )

        dx_center = quantity_cast_to_model_float(self.quantity_factory, dx_center_64)
        self._dxc_64 = dx_center_64
        dy_center = quantity_cast_to_model_float(self.quantity_factory, dy_center_64)
        self._dyc_64 = dy_center_64

        return dx_center, dy_center

    def _compute_dxdy_center_cartesian(self):
        dx_center_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "m",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        dx_center_64.data[:, :] = self._dx_const

        dy_center_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "m",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        dy_center_64.data[:, :] = self._dy_const

        dx_center = quantity_cast_to_model_float(self.quantity_factory, dx_center_64)
        self._dxc_64 = dx_center_64
        dy_center = quantity_cast_to_model_float(self.quantity_factory, dy_center_64)
        self._dyc_64 = dy_center_64

        return dx_center, dy_center

    def _compute_area_cube_sphere(self):
        area_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "m^2",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        area_64.data[:, :] = -1.0e8

        area_64.data[3:-4, 3:-4] = get_area(
            self._grid_64.data[3:-3, 3:-3, 0],
            self._grid_64.data[3:-3, 3:-3, 1],
            RADIUS,
            self._np,
        )
        self._comm.halo_update(area_64, n_points=self._halo)

        return quantity_cast_to_model_float(self.quantity_factory, area_64)

    def _compute_area_cartesian(self):
        area_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "m^2",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        area_64.data[:, :] = self._dx_const * self._dy_const
        return quantity_cast_to_model_float(self.quantity_factory, area_64)

    def _compute_area_c_cube_sphere(self):
        area_cgrid_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM],
            "m^2",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        area_cgrid_64.data[3:-3, 3:-3] = get_area(
            self._agrid_64.data[2:-3, 2:-3, 0],
            self._agrid_64.data[2:-3, 2:-3, 1],
            RADIUS,
            self._np,
        )
        # TODO -- this does not seem to matter? running with or without does
        # not change whether it validates
        set_corner_area_to_triangle_area(
            lon=self._agrid_64.data[2:-3, 2:-3, 0],
            lat=self._agrid_64.data[2:-3, 2:-3, 1],
            area=area_cgrid_64.data[3:-3, 3:-3],
            tile_partitioner=self._tile_partitioner,
            rank=self._rank,
            radius=RADIUS,
            np=self._np,
        )

        set_c_grid_tile_border_area(
            self._dgrid_xyz_64[2:-2, 2:-2, :],
            self._agrid_xyz_64[2:-2, 2:-2, :],
            RADIUS,
            area_cgrid_64.data[3:-3, 3:-3],
            self._tile_partitioner,
            self._rank,
            self._np,
        )
        self._comm.halo_update(area_cgrid_64, n_points=self._halo)

        fill_corners_2d(
            area_cgrid_64.data[:, :, None],
            self._grid_indexing,
            gridtype="B",
            direction="x",
        )
        return quantity_cast_to_model_float(self.quantity_factory, area_cgrid_64)

    def _compute_area_c_cartesian(self):
        area_cgrid_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM],
            "m^2",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        area_cgrid_64.data[:, :] = self._dx_const * self._dy_const
        return quantity_cast_to_model_float(self.quantity_factory, area_cgrid_64)

    def _set_hybrid_pressure_coefficients(self):
        ks = self.quantity_factory.zeros(
            [],
            "",
            dtype=Float,
        )
        ptop = self.quantity_factory.zeros(
            [],
            "Pa",
            dtype=Float,
        )
        ak = self.quantity_factory.zeros(
            [util.Z_INTERFACE_DIM],
            "Pa",
            dtype=Float,
        )
        bk = self.quantity_factory.zeros(
            [util.Z_INTERFACE_DIM],
            "",
            dtype=Float,
        )
        pressure_coefficients = set_hybrid_pressure_coefficients(self._npz)
        ks = pressure_coefficients.ks
        ptop = pressure_coefficients.ptop
        ak.data[:] = asarray(pressure_coefficients.ak, type(ak.data))
        bk.data[:] = asarray(pressure_coefficients.bk, type(bk.data))
        return ks, ptop, ak, bk

    def _calculate_center_vectors_cube_sphere(self):
        ec1_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM, self.CARTESIAN_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        ec2_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM, self.CARTESIAN_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        ec1_64.data[:] = self._np.nan
        ec2_64.data[:] = self._np.nan
        ec1_64.data[:-1, :-1, :3], ec2_64.data[:-1, :-1, :3] = get_center_vector(
            self._dgrid_xyz_64,
            self._grid_type,
            self._halo,
            self._tile_partitioner,
            self._rank,
            self._np,
        )

        ec1 = quantity_cast_to_model_float(self.quantity_factory, ec1_64)
        self._ec1_64 = ec1_64
        ec2 = quantity_cast_to_model_float(self.quantity_factory, ec2_64)
        self._ec2_64 = ec2_64
        return ec1, ec2

    def _calculate_center_vectors_cartesian(self):
        ec1_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM, self.CARTESIAN_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        ec2_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM, self.CARTESIAN_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        ec1_64.data[:, :, 0] = 1.0
        ec2_64.data[:, :, 1] = 1.0

        ec1 = quantity_cast_to_model_float(self.quantity_factory, ec1_64)
        self._ec1_64 = ec1_64
        ec2 = quantity_cast_to_model_float(self.quantity_factory, ec2_64)
        self._ec2_64 = ec2_64
        return ec1, ec2

    def _calculate_vectors_west(self):
        ew1_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM, self.CARTESIAN_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        ew2_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM, self.CARTESIAN_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        ew1_64.data[:] = self._np.nan
        ew2_64.data[:] = self._np.nan
        ew1_64.data[1:-1, :-1, :3], ew2_64.data[1:-1, :-1, :3] = calc_unit_vector_west(
            self._dgrid_xyz_64,
            self._agrid_xyz_64,
            self._grid_type,
            self._halo,
            self._tile_partitioner,
            self._rank,
            self._np,
        )

        ew1 = quantity_cast_to_model_float(self.quantity_factory, ew1_64)
        ew2 = quantity_cast_to_model_float(self.quantity_factory, ew2_64)
        return ew1, ew2

    def _calculate_vectors_west_cartesian(self):
        ew1_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM, self.CARTESIAN_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        ew2_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM, self.CARTESIAN_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        ew1_64.data[:, :, 0] = 1.0
        ew2_64.data[:, :, 1] = 1.0

        ew1 = quantity_cast_to_model_float(self.quantity_factory, ew1_64)
        ew2 = quantity_cast_to_model_float(self.quantity_factory, ew2_64)
        return ew1, ew2

    def _calculate_vectors_south(self):
        es1_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM, self.CARTESIAN_DIM],
            "",
            allow_mismatch_float_precision=True,
        )
        es2_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM, self.CARTESIAN_DIM],
            "",
            allow_mismatch_float_precision=True,
        )
        es1_64.data[:] = self._np.nan
        es2_64.data[:] = self._np.nan
        es1_64.data[:-1, 1:-1, :3], es2_64.data[:-1, 1:-1, :3] = calc_unit_vector_south(
            self._dgrid_xyz_64,
            self._agrid_xyz_64,
            self._grid_type,
            self._halo,
            self._tile_partitioner,
            self._rank,
            self._np,
        )

        es1 = quantity_cast_to_model_float(self.quantity_factory, es1_64)
        es2 = quantity_cast_to_model_float(self.quantity_factory, es2_64)
        return es1, es2

    def _calculate_vectors_south_cartesian(self):
        es1_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM, self.CARTESIAN_DIM],
            "",
            allow_mismatch_float_precision=True,
        )
        es2_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM, self.CARTESIAN_DIM],
            "",
            allow_mismatch_float_precision=True,
        )
        es1_64.data[:, :, 0] = 1.0
        es2_64.data[:, :, 1] = 1.0

        es1 = quantity_cast_to_model_float(self.quantity_factory, es1_64)
        es2 = quantity_cast_to_model_float(self.quantity_factory, es2_64)
        return es1, es2

    def _calculate_more_trig_terms(self, cos_sg, sin_sg):
        cosa_u_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        cosa_v_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        cosa_s_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        sina_u_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        sina_v_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        rsin_u_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        rsin_v_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        rsina_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        rsin2_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        cosa_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        sina_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        (
            cosa_64.data[:, :],
            sina_64.data[:, :],
            cosa_u_64.data[:, :-1],
            cosa_v_64.data[:-1, :],
            cosa_s_64.data[:-1, :-1],
            sina_u_64.data[:, :-1],
            sina_v_64.data[:-1, :],
            rsin_u_64.data[:, :-1],
            rsin_v_64.data[:-1, :],
            rsina_64.data[self._halo : -self._halo, self._halo : -self._halo],
            rsin2_64.data[:-1, :-1],
        ) = calculate_trig_uv(
            self._dgrid_xyz_64,
            cos_sg,
            sin_sg,
            self._halo,
            self._tile_partitioner,
            self._rank,
            self._np,
        )
        return (
            quantity_cast_to_model_float(self.quantity_factory, cosa_64),
            quantity_cast_to_model_float(self.quantity_factory, sina_64),
            quantity_cast_to_model_float(self.quantity_factory, cosa_u_64),
            quantity_cast_to_model_float(self.quantity_factory, cosa_v_64),
            quantity_cast_to_model_float(self.quantity_factory, cosa_s_64),
            quantity_cast_to_model_float(self.quantity_factory, sina_u_64),
            quantity_cast_to_model_float(self.quantity_factory, sina_v_64),
            quantity_cast_to_model_float(self.quantity_factory, rsin_u_64),
            quantity_cast_to_model_float(self.quantity_factory, rsin_v_64),
            quantity_cast_to_model_float(self.quantity_factory, rsina_64),
            quantity_cast_to_model_float(self.quantity_factory, rsin2_64),
        )

    def _init_cell_trigonometry_cube_sphere(self):
        cosa_u_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        cosa_v_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        cosa_s_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        sina_u_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        sina_v_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        rsin_u_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        rsin_v_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        rsina_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        rsin2_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        cosa_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        sina_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )

        # This section calculates the cos_sg and sin_sg terms, which describe the
        # angles of the corners and edges of each cell according to the supergrid:
        #  9---4---8
        #  |       |
        #  1   5   3
        #  |       |
        #  6---2---7

        if self._ec1_64 is None:
            self._ec1, self._ec2 = self._calculate_center_vectors()

        cos_sg, sin_sg = calculate_supergrid_cos_sin(
            self._dgrid_xyz_64,
            self._agrid_xyz_64,
            self._ec1_64.data[:-1, :-1],
            self._ec2_64.data[:-1, :-1],
            self._grid_type,
            self._halo,
            self._tile_partitioner,
            self._rank,
            self._np,
        )

        (
            cosa_64.data[:, :],
            sina_64.data[:, :],
            cosa_u_64.data[:, :-1],
            cosa_v_64.data[:-1, :],
            cosa_s_64.data[:-1, :-1],
            sina_u_64.data[:, :-1],
            sina_v_64.data[:-1, :],
            rsin_u_64.data[:, :-1],
            rsin_v_64.data[:-1, :],
            rsina_64.data[self._halo : -self._halo, self._halo : -self._halo],
            rsin2_64.data[:-1, :-1],
        ) = calculate_trig_uv(
            self._dgrid_xyz_64,
            cos_sg,
            sin_sg,
            self._halo,
            self._tile_partitioner,
            self._rank,
            self._np,
        )

        supergrid_corner_fix(
            cos_sg, sin_sg, self._halo, self._tile_partitioner, self._rank
        )

        supergrid_trig = {}
        for i in range(1, 10):
            supergrid_trig[f"cos_sg{i}"] = self.quantity_factory.zeros(
                [util.X_DIM, util.Y_DIM],
                "",
                dtype=np.float64,
                allow_mismatch_float_precision=True,
            )
            supergrid_trig[f"cos_sg{i}"].data[:-1, :-1] = cos_sg[:, :, i - 1]
            supergrid_trig[f"sin_sg{i}"] = self.quantity_factory.zeros(
                [util.X_DIM, util.Y_DIM],
                "",
                dtype=np.float64,
                allow_mismatch_float_precision=True,
            )
            supergrid_trig[f"sin_sg{i}"].data[:-1, :-1] = sin_sg[:, :, i - 1]

        self._cos_sg1 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["cos_sg1"]
        )
        self._cos_sg2 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["cos_sg2"]
        )
        self._cos_sg3 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["cos_sg3"]
        )
        self._cos_sg4 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["cos_sg4"]
        )
        self._cos_sg5 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["cos_sg5"]
        )
        self._cos_sg6 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["cos_sg6"]
        )
        self._cos_sg7 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["cos_sg7"]
        )
        self._cos_sg8 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["cos_sg8"]
        )
        self._cos_sg9 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["cos_sg9"]
        )
        self._sin_sg1 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["sin_sg1"]
        )
        self._sin_sg2 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["sin_sg2"]
        )
        self._sin_sg3 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["sin_sg3"]
        )
        self._sin_sg4 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["sin_sg4"]
        )
        self._sin_sg5 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["sin_sg5"]
        )
        self._sin_sg5_64 = supergrid_trig["sin_sg5"]
        self._sin_sg6 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["sin_sg6"]
        )
        self._sin_sg7 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["sin_sg7"]
        )
        self._sin_sg8 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["sin_sg8"]
        )
        self._sin_sg9 = quantity_cast_to_model_float(
            self.quantity_factory, supergrid_trig["sin_sg9"]
        )

        # Casting
        self._cosa_u = quantity_cast_to_model_float(self.quantity_factory, cosa_u_64)
        self._cosa_v = quantity_cast_to_model_float(self.quantity_factory, cosa_v_64)
        self._cosa_s = quantity_cast_to_model_float(self.quantity_factory, cosa_s_64)
        self._sina_u = quantity_cast_to_model_float(self.quantity_factory, sina_u_64)
        self._sina_u_64 = sina_u_64
        self._sina_v = quantity_cast_to_model_float(self.quantity_factory, sina_v_64)
        self._sina_v_64 = sina_v_64
        self._rsin_u = quantity_cast_to_model_float(self.quantity_factory, rsin_u_64)
        self._rsin_v = quantity_cast_to_model_float(self.quantity_factory, rsin_v_64)
        self._rsina = quantity_cast_to_model_float(self.quantity_factory, rsina_64)
        self._rsin2 = quantity_cast_to_model_float(self.quantity_factory, rsin2_64)
        self._cosa = quantity_cast_to_model_float(self.quantity_factory, cosa_64)
        self._sina = quantity_cast_to_model_float(self.quantity_factory, sina_64)

    def _init_cell_trigonometry_cartesian(self):

        cosa_u_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        cosa_v_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        cosa_s_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        sina_u_64 = self.quantity_factory.ones(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        sina_v_64 = self.quantity_factory.ones(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        rsin_u_64 = self.quantity_factory.ones(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        rsin_v_64 = self.quantity_factory.ones(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        rsina_64 = self.quantity_factory.ones(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        rsin2_64 = self.quantity_factory.ones(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        cosa_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        sina_64 = self.quantity_factory.ones(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )

        for i in range(1, 10):
            sin_sg = self.quantity_factory.ones(
                [util.X_DIM, util.Y_DIM],
                "",
                dtype=np.float64,
                allow_mismatch_float_precision=True,
            )
            setattr(
                self,
                f"_sin_sg{i}",
                quantity_cast_to_model_float(self.quantity_factory, sin_sg),
            )
            if i == 5:
                self._sin_sg5_64 = sin_sg
            cos_sg = self.quantity_factory.zeros(
                [util.X_DIM, util.Y_DIM],
                "",
                dtype=np.float64,
                allow_mismatch_float_precision=True,
            )
            setattr(
                self,
                f"_cos_sg{i}",
                quantity_cast_to_model_float(self.quantity_factory, cos_sg),
            )

        self._cosa_u = quantity_cast_to_model_float(self.quantity_factory, cosa_u_64)
        self._cosa_v = quantity_cast_to_model_float(self.quantity_factory, cosa_v_64)
        self._cosa_s = quantity_cast_to_model_float(self.quantity_factory, cosa_s_64)
        self._sina_u = quantity_cast_to_model_float(self.quantity_factory, sina_u_64)
        self._sina_u_64 = sina_u_64
        self._sina_v = quantity_cast_to_model_float(self.quantity_factory, sina_v_64)
        self._sina_v_64 = sina_v_64
        self._rsin_u = quantity_cast_to_model_float(self.quantity_factory, rsin_u_64)
        self._rsin_v = quantity_cast_to_model_float(self.quantity_factory, rsin_v_64)
        self._rsina = quantity_cast_to_model_float(self.quantity_factory, rsina_64)
        self._rsin2 = quantity_cast_to_model_float(self.quantity_factory, rsin2_64)
        self._cosa = quantity_cast_to_model_float(self.quantity_factory, cosa_64)
        self._sina = quantity_cast_to_model_float(self.quantity_factory, sina_64)

    def _calculate_derived_trig_terms_for_testing(self):
        """
        As _calculate_derived_trig_terms_for_testing but updates trig attributes
        in-place without the halo updates. For use only in validation tests.
        """
        cosa_u_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        cosa_v_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        cosa_s_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        sina_u_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        sina_v_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        rsin_u_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        rsin_v_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        rsina_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        rsin2_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        cosa_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        sina_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )

        cos_sg = self._np.array(
            [
                self.cos_sg1.data[:-1, :-1],
                self.cos_sg2.data[:-1, :-1],
                self.cos_sg3.data[:-1, :-1],
                self.cos_sg4.data[:-1, :-1],
                self.cos_sg5.data[:-1, :-1],
                self.cos_sg6.data[:-1, :-1],
                self.cos_sg7.data[:-1, :-1],
                self.cos_sg8.data[:-1, :-1],
                self.cos_sg9.data[:-1, :-1],
            ]
        ).transpose([1, 2, 0])
        sin_sg = self._np.array(
            [
                self.sin_sg1.data[:-1, :-1],
                self.sin_sg2.data[:-1, :-1],
                self.sin_sg3.data[:-1, :-1],
                self.sin_sg4.data[:-1, :-1],
                self.sin_sg5.data[:-1, :-1],
                self.sin_sg6.data[:-1, :-1],
                self.sin_sg7.data[:-1, :-1],
                self.sin_sg8.data[:-1, :-1],
                self.sin_sg9.data[:-1, :-1],
            ]
        ).transpose([1, 2, 0])

        (
            cosa_64.data[:, :],
            sina_64.data[:, :],
            cosa_u_64.data[:, :-1],
            cosa_v_64.data[:-1, :],
            cosa_s_64.data[:-1, :-1],
            sina_u_64.data[:, :-1],
            sina_v_64.data[:-1, :],
            rsin_u_64.data[:, :-1],
            rsin_v_64.data[:-1, :],
            rsina_64.data[self._halo : -self._halo, self._halo : -self._halo],
            rsin2_64.data[:-1, :-1],
        ) = calculate_trig_uv(
            self._dgrid_xyz_64,
            cos_sg,
            sin_sg,
            self._halo,
            self._tile_partitioner,
            self._rank,
            self._np,
        )

        self._cosa = quantity_cast_to_model_float(self.quantity_factory, cosa_64)
        self._sina = quantity_cast_to_model_float(self.quantity_factory, sina_64)
        self._cosa_u = quantity_cast_to_model_float(self.quantity_factory, cosa_u_64)
        self._cosa_v = quantity_cast_to_model_float(self.quantity_factory, cosa_v_64)
        self._cosa_s = quantity_cast_to_model_float(self.quantity_factory, cosa_s_64)
        self._sina_u = quantity_cast_to_model_float(self.quantity_factory, sina_u_64)
        self._sina_v = quantity_cast_to_model_float(self.quantity_factory, sina_v_64)
        self._rsin_u = quantity_cast_to_model_float(self.quantity_factory, rsin_u_64)
        self._rsin_v = quantity_cast_to_model_float(self.quantity_factory, rsin_v_64)
        self._rsina = quantity_cast_to_model_float(self.quantity_factory, rsina_64)
        self._rsin2 = quantity_cast_to_model_float(self.quantity_factory, rsin2_64)

    def _calculate_latlon_momentum_correction_cube_sphere(self):
        l2c_v_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=Float,
            allow_mismatch_float_precision=True,
        )
        l2c_u_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=Float,
            allow_mismatch_float_precision=True,
        )
        (
            l2c_v_64.data[self._halo : -self._halo, self._halo : -self._halo - 1],
            l2c_u_64.data[self._halo : -self._halo - 1, self._halo : -self._halo],
        ) = calculate_l2c_vu(self._grid_64.data[:], self._halo, self._np)

        l2c_u = quantity_cast_to_model_float(self.quantity_factory, l2c_u_64)
        l2c_v = quantity_cast_to_model_float(self.quantity_factory, l2c_v_64)

        return l2c_v, l2c_u

    def _calculate_latlon_momentum_correction_cartesian(self):
        l2c_v_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=Float,
            allow_mismatch_float_precision=True,
        )
        l2c_u_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=Float,
            allow_mismatch_float_precision=True,
        )
        l2c_v_64.data[:] = self._np.nan
        l2c_u_64.data[:] = self._np.nan

        l2c_v = quantity_cast_to_model_float(self.quantity_factory, l2c_v_64)
        l2c_u = quantity_cast_to_model_float(self.quantity_factory, l2c_u_64)

        return l2c_v, l2c_u

    def _calculate_xy_unit_vectors_cube_sphere(self):
        ee1_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM, self.CARTESIAN_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        ee2_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM, self.CARTESIAN_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        ee1_64.data[:] = self._np.nan
        ee2_64.data[:] = self._np.nan
        (
            ee1_64.data[self._halo : -self._halo, self._halo : -self._halo, :],
            ee2_64.data[self._halo : -self._halo, self._halo : -self._halo, :],
        ) = calculate_xy_unit_vectors(
            self._dgrid_xyz_64, self._halo, self._tile_partitioner, self._rank, self._np
        )

        ee1 = quantity_cast_to_model_float(self.quantity_factory, ee1_64)
        ee2 = quantity_cast_to_model_float(self.quantity_factory, ee2_64)

        return ee1, ee2

    def _calculate_xy_unit_vectors_cartesian(self):
        ee1_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM, self.CARTESIAN_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        ee2_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_INTERFACE_DIM, self.CARTESIAN_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        ee1_64.data[:] = self._np.nan
        ee2_64.data[:] = self._np.nan

        ee1 = quantity_cast_to_model_float(self.quantity_factory, ee1_64)
        ee2 = quantity_cast_to_model_float(self.quantity_factory, ee2_64)

        return ee1, ee2

    def _calculate_divg_del6(self):
        del6_u_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        del6_v_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        divg_u_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        divg_v_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        sin_sg = [
            self.sin_sg1.data[:-1, :-1],
            self.sin_sg2.data[:-1, :-1],
            self.sin_sg3.data[:-1, :-1],
            self.sin_sg4.data[:-1, :-1],
            self.sin_sg5.data[:-1, :-1],
        ]
        sin_sg = self._np.array(sin_sg).transpose(1, 2, 0)
        if self._sina_u_64 is None:
            self._init_cell_trigonometry()
        if self._dx_64 is None:
            self._dx, self._dy = self._compute_dxdy()
        if self._dxc_64 is None:
            self._dx_center, self._dy_center = self._compute_dxdy_center()
        (
            divg_u_64.data[:-1, :],
            divg_v_64.data[:, :-1],
            del6_u_64.data[:-1, :],
            del6_v_64.data[:, :-1],
        ) = calculate_divg_del6(
            sin_sg,
            self._sina_u_64.data[:, :-1],
            self._sina_v_64.data[:-1, :],
            self._dx_64.data[:-1, :],
            self._dy_64.data[:, :-1],
            self._dxc_64.data[:, :-1],
            self._dyc_64.data[:-1, :],
            self._halo,
            self._tile_partitioner,
            self._rank,
        )

        if self._grid_type < 3:
            self._comm.vector_halo_update(divg_v_64, divg_u_64, n_points=self._halo)
            self._comm.vector_halo_update(del6_v_64, del6_u_64, n_points=self._halo)
            # TODO: Add support for unsigned vector halo updates
            # instead of handling ad-hoc here
            divg_v_64.data[divg_v_64.data < 0] *= -1
            divg_u_64.data[divg_u_64.data < 0] *= -1
            del6_v_64.data[del6_v_64.data < 0] *= -1
            del6_u_64.data[del6_u_64.data < 0] *= -1

        divg_v = quantity_cast_to_model_float(self.quantity_factory, divg_v_64)
        divg_u = quantity_cast_to_model_float(self.quantity_factory, divg_u_64)
        del6_v = quantity_cast_to_model_float(self.quantity_factory, del6_v_64)
        del6_u = quantity_cast_to_model_float(self.quantity_factory, del6_u_64)

        return del6_u, del6_v, divg_u, divg_v

    def _calculate_divg_del6_nohalos_for_testing(self):
        """
        As _calculate_divg_del6 but updates self.divg and self.del6 attributes
        in-place without the halo updates. For use only in validation tests.
        """
        del6_u_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        del6_v_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        divg_u_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        divg_v_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        sin_sg = [
            self.sin_sg1.data[:-1, :-1],
            self.sin_sg2.data[:-1, :-1],
            self.sin_sg3.data[:-1, :-1],
            self.sin_sg4.data[:-1, :-1],
            self.sin_sg5.data[:-1, :-1],
        ]
        sin_sg = self._np.array(sin_sg).transpose(1, 2, 0)
        if self._sina_u_64 is None:
            self._init_cell_trigonometry()
        if self._dx_64 is None:
            self._dx, self._dy = self._compute_dxdy()
        if self._dxc_64 is None:
            self._dx_center, self._dy_center = self._compute_dxdy_center()
        (
            divg_u_64.data[:-1, :],
            divg_v_64.data[:, :-1],
            del6_u_64.data[:-1, :],
            del6_v_64.data[:, :-1],
        ) = calculate_divg_del6(
            sin_sg,
            self._sina_u_64.data[:, :-1],
            self._sina_v_64.data[:-1, :],
            self._dx_64.data[:-1, :],
            self._dy_64.data[:, :-1],
            self._dxc_64.data[:, :-1],
            self._dyc_64.data[:-1, :],
            self._halo,
            self._tile_partitioner,
            self._rank,
        )
        self._divg_v = quantity_cast_to_model_float(self.quantity_factory, divg_v_64)
        self._divg_u = quantity_cast_to_model_float(self.quantity_factory, divg_u_64)
        self._del6_v = quantity_cast_to_model_float(self.quantity_factory, del6_v_64)
        self._del6_u = quantity_cast_to_model_float(self.quantity_factory, del6_u_64)

    def _calculate_unit_vectors_lonlat_cube_sphere(self):
        vlon_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM, self.CARTESIAN_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        vlat_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM, self.CARTESIAN_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )

        vlon_64.data[:-1, :-1], vlat_64.data[:-1, :-1] = unit_vector_lonlat(
            self._agrid_64.data[:-1, :-1], self._np
        )

        vlon = quantity_cast_to_model_float(self.quantity_factory, vlon_64)
        self._vlon_64 = vlon_64
        vlat = quantity_cast_to_model_float(self.quantity_factory, vlat_64)
        self._vlat_64 = vlat_64

        return vlon, vlat

    def _calculate_unit_vectors_lonlat_cartesian(self):
        vlon_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM, self.CARTESIAN_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        vlat_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM, self.CARTESIAN_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        vlon_64.data[:] = self._np.nan
        vlat_64.data[:] = self._np.nan

        vlon = quantity_cast_to_model_float(self.quantity_factory, vlon_64)
        self._vlon_64 = vlon_64
        vlat = quantity_cast_to_model_float(self.quantity_factory, vlat_64)
        self._vlat_64 = vlat_64

        return vlon, vlat

    def _calculate_grid_z(self):
        z11_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        z12_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        z21_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        z22_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )

        if self._ec1_64 is None:
            self._ec1, self._ec2 = self._calculate_center_vectors()
        if self._vlon_64 is None:
            self._vlon, self._vlat = self._calculate_unit_vectors_lonlat()

        (
            z11_64.data[:-1, :-1],
            z12_64.data[:-1, :-1],
            z21_64.data[:-1, :-1],
            z22_64.data[:-1, :-1],
        ) = calculate_grid_z(
            self._ec1_64.data[:-1, :-1],
            self._ec2_64.data[:-1, :-1],
            self._vlon_64.data[:-1, :-1],
            self._vlat_64.data[:-1, :-1],
            self._np,
        )

        z11 = quantity_cast_to_model_float(self.quantity_factory, z11_64)
        self._z11_64 = z11_64
        z12 = quantity_cast_to_model_float(self.quantity_factory, z12_64)
        self._z12_64 = z12_64
        z21 = quantity_cast_to_model_float(self.quantity_factory, z21_64)
        self._z21_64 = z21_64
        z22 = quantity_cast_to_model_float(self.quantity_factory, z22_64)
        self._z22_64 = z22_64

        return z11, z12, z21, z22

    def _calculate_grid_a(self):
        a11_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        a12_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        a21_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        a22_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )

        if self._z11_64 is None:
            self._z11, self._z12, self._z21, self._z22 = self._calculate_grid_z()
        if self._sin_sg5_64 is None:
            self._init_cell_trigonometry()

        (
            a11_64.data[:-1, :-1],
            a12_64.data[:-1, :-1],
            a21_64.data[:-1, :-1],
            a22_64.data[:-1, :-1],
        ) = calculate_grid_a(
            self._z11_64.data[:-1, :-1],
            self._z12_64.data[:-1, :-1],
            self._z21_64.data[:-1, :-1],
            self._z22_64.data[:-1, :-1],
            self._sin_sg5_64.data[:-1, :-1],
        )

        a11 = quantity_cast_to_model_float(self.quantity_factory, a11_64)
        a12 = quantity_cast_to_model_float(self.quantity_factory, a12_64)
        a21 = quantity_cast_to_model_float(self.quantity_factory, a21_64)
        a22 = quantity_cast_to_model_float(self.quantity_factory, a22_64)

        return a11, a12, a21, a22

    def _calculate_edge_factors(self):
        nhalo = self._halo
        edge_s_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        edge_n_64 = self.quantity_factory.zeros(
            [util.X_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        edge_e_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        edge_w_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_INTERFACE_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        (
            edge_w_64.data[:, nhalo:-nhalo],
            edge_e_64.data[:, nhalo:-nhalo],
            edge_s_64.data[nhalo:-nhalo],
            edge_n_64.data[nhalo:-nhalo],
        ) = edge_factors(
            self.gridvar,
            self._agrid_64.data[:-1, :-1],
            self._grid_type,
            nhalo,
            self._tile_partitioner,
            self._rank,
            RADIUS,
            self._np,
        )

        edge_w = quantity_cast_to_model_float(self.quantity_factory, edge_w_64)
        edge_e = quantity_cast_to_model_float(self.quantity_factory, edge_e_64)
        edge_s = quantity_cast_to_model_float(self.quantity_factory, edge_s_64)
        edge_n = quantity_cast_to_model_float(self.quantity_factory, edge_n_64)

        return edge_w, edge_e, edge_s, edge_n

    def _calculate_edge_a2c_vect_factors(self):
        edge_vect_s_64 = self.quantity_factory.zeros(
            [util.X_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        edge_vect_n_64 = self.quantity_factory.zeros(
            [util.X_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        edge_vect_e_64 = self.quantity_factory.zeros(
            [util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        edge_vect_w_64 = self.quantity_factory.zeros(
            [util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        (
            edge_vect_w_64.data[:-1],
            edge_vect_e_64.data[:-1],
            edge_vect_s_64.data[:-1],
            edge_vect_n_64.data[:-1],
        ) = efactor_a2c_v(
            self.gridvar,
            self._agrid_64.data[:-1, :-1],
            self._grid_type,
            self._halo,
            self._tile_partitioner,
            self._rank,
            RADIUS,
            self._np,
        )

        edge_vect_w = quantity_cast_to_model_float(
            self.quantity_factory, edge_vect_w_64
        )
        edge_vect_e = quantity_cast_to_model_float(
            self.quantity_factory, edge_vect_e_64
        )
        edge_vect_s = quantity_cast_to_model_float(
            self.quantity_factory, edge_vect_s_64
        )
        edge_vect_n = quantity_cast_to_model_float(
            self.quantity_factory, edge_vect_n_64
        )
        return edge_vect_w, edge_vect_e, edge_vect_s, edge_vect_n

    def _calculate_2d_edge_a2c_vect_factors(self):
        edge_vect_e_2d_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        edge_vect_w_2d_64 = self.quantity_factory.zeros(
            [util.X_DIM, util.Y_DIM],
            "",
            dtype=np.float64,
            allow_mismatch_float_precision=True,
        )
        shape = self.lon.data.shape
        east_edge_data = self.edge_vect_e_1d.data[self._np.newaxis, ...]
        east_edge_data = self._np.repeat(east_edge_data, shape[0], axis=0)
        west_edge_data = self.edge_vect_w_1d.data[self._np.newaxis, ...]
        west_edge_data = self._np.repeat(west_edge_data, shape[0], axis=0)
        edge_vect_e_2d_64.data[:-1, :-1], edge_vect_w_2d_64.data[:-1, :-1] = (
            east_edge_data[:-1, :-1],
            west_edge_data[:-1, :-1],
        )

        edge_vect_e_2d = quantity_cast_to_model_float(
            self.quantity_factory, edge_vect_e_2d_64
        )
        edge_vect_w_2d = quantity_cast_to_model_float(
            self.quantity_factory, edge_vect_w_2d_64
        )
        return edge_vect_e_2d, edge_vect_w_2d

    def _reduce_global_area_minmaxes(self):
        min_area = self._np.min(self.area.data[3:-4, 3:-4])[()]
        max_area = self._np.max(self.area.data[3:-4, 3:-4])[()]
        min_area_c = self._np.min(self.area_c.data[3:-4, 3:-4])[()]
        max_area_c = self._np.max(self.area_c.data[3:-4, 3:-4])[()]
        self._da_min = float(self._comm.comm.allreduce(min_area, min))
        self._da_max = float(self._comm.comm.allreduce(max_area, max))
        self._da_min_c = float(self._comm.comm.allreduce(min_area_c, min))
        self._da_max_c = float(self._comm.comm.allreduce(max_area_c, max))
