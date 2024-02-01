import dataclasses
import pathlib

import xarray as xr


# TODO: if we can remove translate tests in favor of checkpointer tests,
# we can remove this "disallowed" import (ndsl.util does not depend on ndsl.dsl)
try:
    from ndsl.dsl.gt4py_utils import split_cartesian_into_storages
except ImportError:
    split_cartesian_into_storages = None
import ndsl.constants as constants
from ndsl.constants import Z_DIM, Z_INTERFACE_DIM
from ndsl.filesystem import get_fs
from ndsl.initialization import QuantityFactory
from ndsl.quantity import Quantity

from .generation import MetricTerms


@dataclasses.dataclass(frozen=True)
class DampingCoefficients:
    """
    Terms used to compute damping coefficients.
    """

    divg_u: Quantity
    divg_v: Quantity
    del6_u: Quantity
    del6_v: Quantity
    da_min: float
    da_min_c: float

    @classmethod
    def new_from_metric_terms(cls, metric_terms: MetricTerms):
        return cls(
            divg_u=metric_terms.divg_u,
            divg_v=metric_terms.divg_v,
            del6_u=metric_terms.del6_u,
            del6_v=metric_terms.del6_v,
            da_min=metric_terms.da_min,
            da_min_c=metric_terms.da_min_c,
        )


@dataclasses.dataclass(frozen=True)
class HorizontalGridData:
    """
    Terms defining the horizontal grid.
    """

    lon: Quantity
    lat: Quantity
    lon_agrid: Quantity
    lat_agrid: Quantity
    area: Quantity
    area_64: Quantity
    rarea: Quantity
    # TODO: refactor this to "area_c" and invert where used
    rarea_c: Quantity
    dx: Quantity
    dy: Quantity
    dxc: Quantity
    dyc: Quantity
    dxa: Quantity
    dya: Quantity
    # TODO: refactor usages to invert "normal" versions instead
    rdx: Quantity
    rdy: Quantity
    rdxc: Quantity
    rdyc: Quantity
    rdxa: Quantity
    rdya: Quantity
    ee1: Quantity
    ee2: Quantity
    es1: Quantity
    ew2: Quantity
    a11: Quantity
    a12: Quantity
    a21: Quantity
    a22: Quantity
    edge_w: Quantity
    edge_e: Quantity
    edge_s: Quantity
    edge_n: Quantity

    @classmethod
    def new_from_metric_terms(cls, metric_terms: MetricTerms) -> "HorizontalGridData":
        return cls(
            lon=metric_terms.lon,
            lat=metric_terms.lat,
            lon_agrid=metric_terms.lon_agrid,
            lat_agrid=metric_terms.lat_agrid,
            area=metric_terms.area,
            area_64=metric_terms.area,
            rarea=metric_terms.rarea,
            rarea_c=metric_terms.rarea_c,
            dx=metric_terms.dx,
            dy=metric_terms.dy,
            dxc=metric_terms.dxc,
            dyc=metric_terms.dyc,
            dxa=metric_terms.dxa,
            dya=metric_terms.dya,
            rdx=metric_terms.rdx,
            rdy=metric_terms.rdy,
            rdxc=metric_terms.rdxc,
            rdyc=metric_terms.rdyc,
            rdxa=metric_terms.rdxa,
            rdya=metric_terms.rdya,
            ee1=metric_terms.ee1,
            ee2=metric_terms.ee2,
            es1=metric_terms.es1,
            ew2=metric_terms.ew2,
            a11=metric_terms.a11,
            a12=metric_terms.a12,
            a21=metric_terms.a21,
            a22=metric_terms.a22,
            edge_w=metric_terms.edge_w,
            edge_e=metric_terms.edge_e,
            edge_s=metric_terms.edge_s,
            edge_n=metric_terms.edge_n,
        )


@dataclasses.dataclass
class VerticalGridData:
    """
    Terms defining the vertical grid.

    Eulerian vertical grid is defined by p = ak + bk * p_ref
    """

    # TODO: make these non-optional, make FloatFieldK a true type and use it
    ak: Quantity
    bk: Quantity
    """
    reference pressure (Pa) used to define pressure at vertical interfaces,
    where p = ak + bk * p_ref
    """

    def __post_init__(self):
        self._dp_ref = None
        self._p = None
        self._p_interface = None

    @classmethod
    def new_from_metric_terms(cls, metric_terms: MetricTerms) -> "VerticalGridData":
        return cls(
            ak=metric_terms.ak,
            bk=metric_terms.bk,
        )

    @classmethod
    def from_restart(cls, restart_path: str, quantity_factory: QuantityFactory):
        fs = get_fs(restart_path)
        restart_files = fs.ls(restart_path)
        data_file = restart_files[
            [fname.endswith("fv_core.res.nc") for fname in restart_files].index(True)
        ]

        ak_bk_data_file = pathlib.Path(restart_path) / data_file
        if not fs.isfile(ak_bk_data_file):
            raise ValueError(
                """vertical_grid_from_restart is true,
                but no fv_core.res.nc in restart data file."""
            )

        ak = quantity_factory.zeros([Z_INTERFACE_DIM], units="Pa")
        bk = quantity_factory.zeros([Z_INTERFACE_DIM], units="")
        with fs.open(ak_bk_data_file, "rb") as f:
            ds = xr.open_dataset(f).isel(Time=0).drop_vars("Time")
            ak.view[:] = ds["ak"].values
            bk.view[:] = ds["bk"].values

        return cls(ak=ak, bk=bk)

    @property
    def p_ref(self) -> float:
        """
        reference pressure (Pa)
        """
        return 1e5

    @property
    def p_interface(self) -> Quantity:
        if self._p_interface is None:
            p_interface_data = self.ak.view[:] + self.bk.view[:] * self.p_ref
            self._p_interface = Quantity(
                p_interface_data,
                dims=[Z_INTERFACE_DIM],
                units="Pa",
                gt4py_backend=self.ak.gt4py_backend,
            )
        return self._p_interface

    @property
    def p(self) -> Quantity:
        if self._p is None:
            p_data = (
                self.p_interface.view[1:] - self.p_interface.view[:-1]
            ) / self.p_interface.np.log(
                self.p_interface.view[1:] / self.p_interface.view[:-1]
            )
            self._p = Quantity(
                p_data,
                dims=[Z_DIM],
                units="Pa",
                gt4py_backend=self.p_interface.gt4py_backend,
            )
        return self._p

    @property
    def dp(self) -> Quantity:
        if self._dp_ref is None:
            dp_ref_data = (
                self.ak.view[1:]
                - self.ak.view[:-1]
                + (self.bk.view[1:] - self.bk.view[:-1]) * self.p_ref
            )
            self._dp_ref = Quantity(
                dp_ref_data,
                dims=[Z_DIM],
                units="Pa",
                gt4py_backend=self.ak.gt4py_backend,
            )
        return self._dp_ref

    @property
    def ptop(self) -> float:
        """
        top of atmosphere pressure (Pa)
        """
        if self.bk.view[0] != 0:
            raise ValueError("ptop is not well-defined when top-of-atmosphere bk != 0")
        return float(self.ak.view[0])


@dataclasses.dataclass(frozen=True)
class ContravariantGridData:
    """
    Grid variables used for converting vectors from covariant to
    contravariant components.
    """

    cosa: Quantity
    cosa_u: Quantity
    cosa_v: Quantity
    cosa_s: Quantity
    sina_u: Quantity
    sina_v: Quantity
    rsina: Quantity
    rsin_u: Quantity
    rsin_v: Quantity
    rsin2: Quantity

    @classmethod
    def new_from_metric_terms(
        cls, metric_terms: MetricTerms
    ) -> "ContravariantGridData":
        return cls(
            cosa=metric_terms.cosa,
            cosa_u=metric_terms.cosa_u,
            cosa_v=metric_terms.cosa_v,
            cosa_s=metric_terms.cosa_s,
            sina_u=metric_terms.sina_u,
            sina_v=metric_terms.sina_v,
            rsina=metric_terms.rsina,
            rsin_u=metric_terms.rsin_u,
            rsin_v=metric_terms.rsin_v,
            rsin2=metric_terms.rsin2,
        )


@dataclasses.dataclass(frozen=True)
class AngleGridData:
    """
    sin and cos of certain angles used in metric calculations.

    Corresponds in the fortran code to sin_sg and cos_sg.
    """

    sin_sg1: Quantity
    sin_sg2: Quantity
    sin_sg3: Quantity
    sin_sg4: Quantity
    cos_sg1: Quantity
    cos_sg2: Quantity
    cos_sg3: Quantity
    cos_sg4: Quantity

    @classmethod
    def new_from_metric_terms(cls, metric_terms: MetricTerms) -> "AngleGridData":
        return cls(
            sin_sg1=metric_terms.sin_sg1,
            sin_sg2=metric_terms.sin_sg2,
            sin_sg3=metric_terms.sin_sg3,
            sin_sg4=metric_terms.sin_sg4,
            cos_sg1=metric_terms.cos_sg1,
            cos_sg2=metric_terms.cos_sg2,
            cos_sg3=metric_terms.cos_sg3,
            cos_sg4=metric_terms.cos_sg4,
        )


class GridData:
    # TODO: add docstrings to remaining properties

    def __init__(
        self,
        horizontal_data: HorizontalGridData,
        vertical_data: VerticalGridData,
        contravariant_data: ContravariantGridData,
        angle_data: AngleGridData,
    ):
        self._horizontal_data = horizontal_data
        self._vertical_data = vertical_data
        self._contravariant_data = contravariant_data
        self._angle_data = angle_data
        self._fC = None
        self._fC_agrid = None

    @classmethod
    def new_from_metric_terms(cls, metric_terms: MetricTerms):
        horizontal_data = HorizontalGridData.new_from_metric_terms(metric_terms)
        vertical_data = VerticalGridData.new_from_metric_terms(metric_terms)
        contravariant_data = ContravariantGridData.new_from_metric_terms(metric_terms)
        angle_data = AngleGridData.new_from_metric_terms(metric_terms)
        return cls(horizontal_data, vertical_data, contravariant_data, angle_data)

    @property
    def lon(self):
        """longitude of cell corners"""
        return self._horizontal_data.lon

    @property
    def lat(self):
        """latitude of cell corners"""
        return self._horizontal_data.lat

    @property
    def lon_agrid(self) -> Quantity:
        """longitude on the A-grid (cell centers)"""
        return self._horizontal_data.lon_agrid

    @property
    def lat_agrid(self) -> Quantity:
        """latitude on the A-grid (cell centers)"""
        return self._horizontal_data.lat_agrid

    @staticmethod
    def _fC_from_lat(lat: Quantity) -> Quantity:
        np = lat.np
        data = 2.0 * constants.OMEGA * np.sin(lat.data)
        return Quantity(
            data,
            units="1/s",
            dims=lat.dims,
            origin=lat.origin,
            extent=lat.extent,
            gt4py_backend=lat.gt4py_backend,
        )

    @property
    def fC(self):
        """Coriolis parameter at cell corners"""
        if self._fC is None:
            self._fC = self._fC_from_lat(self.lat)
        return self._fC

    @property
    def fC_agrid(self):
        """Coriolis parameter at cell centers"""
        if self._fC_agrid is None:
            self._fC_agrid = self._fC_from_lat(self.lat_agrid)
        return self._fC_agrid

    @property
    def area(self):
        """Gridcell area"""
        return self._horizontal_data.area

    @property
    def area_64(self):
        """Gridcell area (64-bit)"""
        return self._horizontal_data.area_64

    @property
    def rarea(self):
        """1 / area"""
        return self._horizontal_data.rarea

    @property
    def rarea_c(self):
        return self._horizontal_data.rarea_c

    @property
    def dx(self):
        """distance between cell corners in x-direction"""
        return self._horizontal_data.dx

    @property
    def dy(self):
        """distance between cell corners in y-direction"""
        return self._horizontal_data.dy

    @property
    def dxc(self):
        """distance between gridcell centers in x-direction"""
        return self._horizontal_data.dxc

    @property
    def dyc(self):
        """distance between gridcell centers in y-direction"""
        return self._horizontal_data.dyc

    @property
    def dxa(self):
        """distance between centers of west and east edges of gridcell"""
        return self._horizontal_data.dxa

    @property
    def dya(self):
        """distance between centers of north and south edges of gridcell"""
        return self._horizontal_data.dya

    @property
    def rdx(self):
        """1 / dx"""
        return self._horizontal_data.rdx

    @property
    def rdy(self):
        """1 / dy"""
        return self._horizontal_data.rdy

    @property
    def rdxc(self):
        """1 / dxc"""
        return self._horizontal_data.rdxc

    @property
    def rdyc(self):
        """1 / dyc"""
        return self._horizontal_data.rdyc

    @property
    def rdxa(self):
        """1 / dxa"""
        return self._horizontal_data.rdxa

    @property
    def rdya(self):
        """1 / dya"""
        return self._horizontal_data.rdya

    @property
    def ee1(self) -> Quantity:
        return self._horizontal_data.ee1

    @property
    def ee2(self) -> Quantity:
        return self._horizontal_data.ee2

    @property
    def es1(self) -> Quantity:
        return self._horizontal_data.es1

    @property
    def ew2(self) -> Quantity:
        return self._horizontal_data.ew2

    @property
    def a11(self):
        return self._horizontal_data.a11

    @property
    def a12(self):
        return self._horizontal_data.a12

    @property
    def a21(self):
        return self._horizontal_data.a21

    @property
    def a22(self):
        return self._horizontal_data.a22

    @property
    def edge_w(self):
        return self._horizontal_data.edge_w

    @property
    def edge_e(self):
        return self._horizontal_data.edge_e

    @property
    def edge_s(self):
        return self._horizontal_data.edge_s

    @property
    def edge_n(self):
        return self._horizontal_data.edge_n

    @property
    def p_ref(self) -> float:
        """
        reference pressure (Pa) used to define pressure at vertical interfaces,
        where p = ak + bk * p_ref
        """
        return self._vertical_data.p_ref

    @property
    def p(self) -> Quantity:
        """
        Reference pressure profile for Eulerian grid, defined at cell centers.
        """
        return self._vertical_data.p

    @property
    def ak(self) -> Quantity:
        """
        constant used to define pressure at vertical interfaces,
        where p = ak + bk * p_ref
        """
        return self._vertical_data.ak

    @ak.setter
    def ak(self, value: Quantity):
        self._vertical_data.ak = value

    @property
    def bk(self) -> Quantity:
        """
        constant used to define pressure at vertical interfaces,
        where p = ak + bk * p_ref
        """
        return self._vertical_data.bk

    @bk.setter
    def bk(self, value: Quantity):
        self._vertical_data.bk = value

    @property
    def ks(self):
        return self._vertical_data.ks

    @ks.setter
    def ks(self, value):
        self._vertical_data.ks = value

    @property
    def ptop(self):
        """pressure at top of atmosphere (Pa)"""
        return self._vertical_data.ptop

    @ptop.setter
    def ptop(self, value):
        self._vertical_data.ptop = value

    @property
    def dp_ref(self) -> Quantity:
        return self._vertical_data.dp

    @property
    def cosa(self):
        return self._contravariant_data.cosa

    @property
    def cosa_u(self):
        return self._contravariant_data.cosa_u

    @property
    def cosa_v(self):
        return self._contravariant_data.cosa_v

    @property
    def cosa_s(self):
        return self._contravariant_data.cosa_s

    @property
    def sina_u(self):
        return self._contravariant_data.sina_u

    @property
    def sina_v(self):
        return self._contravariant_data.sina_v

    @property
    def rsina(self):
        return self._contravariant_data.rsina

    @property
    def rsin_u(self):
        return self._contravariant_data.rsin_u

    @property
    def rsin_v(self):
        return self._contravariant_data.rsin_v

    @property
    def rsin2(self):
        return self._contravariant_data.rsin2

    @property
    def sin_sg1(self):
        return self._angle_data.sin_sg1

    @property
    def sin_sg2(self):
        return self._angle_data.sin_sg2

    @property
    def sin_sg3(self):
        return self._angle_data.sin_sg3

    @property
    def sin_sg4(self):
        return self._angle_data.sin_sg4

    @property
    def cos_sg1(self):
        return self._angle_data.cos_sg1

    @property
    def cos_sg2(self):
        return self._angle_data.cos_sg2

    @property
    def cos_sg3(self):
        return self._angle_data.cos_sg3

    @property
    def cos_sg4(self):
        return self._angle_data.cos_sg4


@dataclasses.dataclass(frozen=True)
class DriverGridData:
    """
    Terms used to Apply Physics changes to the Dycore.
    Attributes:
      vlon1: x-component of unit lon vector in eastward longitude direction
      vlon2: y-component of unit lon vector in eastward longitude direction
      vlon3: z-component of unit lon vector in eastward longitude direction
      vlat1: x-component of unit lat vector in northward latitude direction
      vlat2: y-component of unit lat vector in northward latitude direction
      vlat3: z-component of unit lat vector in northward latitude direction
      edge_vect_w: factor to interpolate A to C grids at the western grid edge
      edge_vect_e: factor to interpolate A to C grids at the easter grid edge
      edge_vect_s: factor to interpolate A to C grids at the southern grid edge
      edge_vect_n: factor to interpolate A to C grids at the northern grid edge
      es1_1: x-component of grid local unit vector in x-direction at cell edge
      es1_2: y-component of grid local unit vector in x-direction at cell edge
      es1_3: z-component of grid local unit vector in x-direction at cell edge
      ew2_1: x-component of grid local unit vector in y-direction at cell edge
      ew2_2: y-component of grid local unit vector in y-direction at cell edge
      ew2_3: z-component of grid local unit vector in y-direction at cell edge
    """

    vlon1: Quantity
    vlon2: Quantity
    vlon3: Quantity
    vlat1: Quantity
    vlat2: Quantity
    vlat3: Quantity
    edge_vect_w: Quantity
    edge_vect_e: Quantity
    edge_vect_s: Quantity
    edge_vect_n: Quantity
    es1_1: Quantity
    es1_2: Quantity
    es1_3: Quantity
    ew2_1: Quantity
    ew2_2: Quantity
    ew2_3: Quantity
    grid_type: int

    @classmethod
    def new_from_metric_terms(cls, metric_terms: MetricTerms) -> "DriverGridData":
        return cls.new_from_grid_variables(
            vlon=metric_terms.vlon,
            vlat=metric_terms.vlon,
            edge_vect_n=metric_terms.edge_vect_n,
            edge_vect_s=metric_terms.edge_vect_s,
            edge_vect_e=metric_terms.edge_vect_e,
            edge_vect_w=metric_terms.edge_vect_w,
            es1=metric_terms.es1,
            ew2=metric_terms.ew2,
            grid_type=metric_terms._grid_type,
        )

    @classmethod
    def new_from_grid_variables(
        cls,
        vlon: Quantity,
        vlat: Quantity,
        edge_vect_n: Quantity,
        edge_vect_s: Quantity,
        edge_vect_e: Quantity,
        edge_vect_w: Quantity,
        es1: Quantity,
        ew2: Quantity,
        grid_type: int = 0,
    ) -> "DriverGridData":
        try:
            vlon1, vlon2, vlon3 = split_quantity_along_last_dim(vlon)
            vlat1, vlat2, vlat3 = split_quantity_along_last_dim(vlat)
            es1_1, es1_2, es1_3 = split_quantity_along_last_dim(es1)
            ew2_1, ew2_2, ew2_3 = split_quantity_along_last_dim(ew2)
        except (AttributeError, TypeError):
            vlon1, vlon2, vlon3 = split_cartesian_into_storages(vlon)
            vlat1, vlat2, vlat3 = split_cartesian_into_storages(vlat)
            es1_1, es1_2, es1_3 = split_cartesian_into_storages(es1)
            ew2_1, ew2_2, ew2_3 = split_cartesian_into_storages(ew2)

        return cls(
            vlon1=vlon1,
            vlon2=vlon2,
            vlon3=vlon3,
            vlat1=vlat1,
            vlat2=vlat2,
            vlat3=vlat3,
            es1_1=es1_1,
            es1_2=es1_2,
            es1_3=es1_3,
            ew2_1=ew2_1,
            ew2_2=ew2_2,
            ew2_3=ew2_3,
            edge_vect_w=edge_vect_w,
            edge_vect_e=edge_vect_e,
            edge_vect_s=edge_vect_s,
            edge_vect_n=edge_vect_n,
            grid_type=grid_type,
        )


def split_quantity_along_last_dim(quantity):
    """Split a quantity along the last dimension into a list of quantities.

    Args:
        quantity: Quantity to split.

    Returns:
        List of quantities.
    """
    return_list = []
    for i in range(quantity.data.shape[-1]):
        return_list.append(
            Quantity(
                data=quantity.data[..., i],
                dims=quantity.dims[:-1],
                units=quantity.units,
                origin=quantity.origin[:-1],
                extent=quantity.extent[:-1],
                gt4py_backend=quantity.gt4py_backend,
            )
        )
    return return_list
