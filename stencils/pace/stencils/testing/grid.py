from typing import Tuple

import numpy as np

import pace.util
from pace.dsl import gt4py_utils as utils
from pace.dsl.stencil import GridIndexing
from pace.util.grid import (
    AngleGridData,
    ContravariantGridData,
    DampingCoefficients,
    GridData,
    HorizontalGridData,
    MetricTerms,
    VerticalGridData,
)
from pace.util.halo_data_transformer import QuantityHaloSpec


TRACER_DIM = "tracers"


class Grid:
    # indices = ["is_", "ie", "isd", "ied", "js", "je", "jsd", "jed"]
    index_pairs = [("is_", "js"), ("ie", "je"), ("isd", "jsd"), ("ied", "jed")]
    shape_params = ["npz", "npx", "npy"]
    # npx -- number of grid corners on one tile of the domain
    # grid.ie == npx - 1identified east edge in fortran
    # But we need to add the halo - 1 to change this check to 0 based python arrays
    # grid.ie == npx + halo - 2

    def __init__(
        self,
        indices,
        shape_params,
        rank,
        layout,
        backend,
        data_fields={},
        local_indices=False,
    ):
        self.rank = rank
        self.backend = backend
        self.partitioner = pace.util.TilePartitioner(layout)
        self.subtile_index = self.partitioner.subtile_index(self.rank)
        self.layout = layout
        for s in self.shape_params:
            setattr(self, s, int(shape_params[s]))
        self.subtile_width_x = int((self.npx - 1) / self.layout[0])
        self.subtile_width_y = int((self.npy - 1) / self.layout[1])
        for ivar, jvar in self.index_pairs:
            local_i, local_j = int(indices[ivar]), int(indices[jvar])
            if not local_indices:
                local_i, local_j = self.global_to_local_indices(local_i, local_j)
            setattr(self, ivar, local_i)
            setattr(self, jvar, local_j)
        self.nid = int(self.ied - self.isd + 1)
        self.njd = int(self.jed - self.jsd + 1)
        self.nic = int(self.ie - self.is_ + 1)
        self.njc = int(self.je - self.js + 1)
        self.halo = utils.halo
        self.global_is, self.global_js = self.local_to_global_indices(self.is_, self.js)
        self.global_ie, self.global_je = self.local_to_global_indices(self.ie, self.je)
        self.global_isd, self.global_jsd = self.local_to_global_indices(
            self.isd, self.jsd
        )
        self.global_ied, self.global_jed = self.local_to_global_indices(
            self.ied, self.jed
        )
        self.west_edge = self.global_is == self.halo
        self.east_edge = self.global_ie == self.npx + self.halo - 2
        self.south_edge = self.global_js == self.halo
        self.north_edge = self.global_je == self.npy + self.halo - 2

        self.j_offset = self.js - self.jsd - 1
        self.i_offset = self.is_ - self.isd - 1
        self.sw_corner = self.west_edge and self.south_edge
        self.se_corner = self.east_edge and self.south_edge
        self.nw_corner = self.west_edge and self.north_edge
        self.ne_corner = self.east_edge and self.north_edge
        self.data_fields = {}
        self.add_data(data_fields)
        self._sizer = None
        self._quantity_factory = None
        self._grid_data = None
        self._damping_coefficients = None

    @property
    def sizer(self):
        if self._sizer is None:
            # in the future this should use from_namelist, when we have a non-flattened
            # namelist
            self._sizer = pace.util.SubtileGridSizer.from_tile_params(
                nx_tile=self.npx - 1,
                ny_tile=self.npy - 1,
                nz=self.npz,
                n_halo=self.halo,
                extra_dim_lengths={
                    MetricTerms.LON_OR_LAT_DIM: 2,
                    MetricTerms.TILE_DIM: 6,
                    MetricTerms.CARTESIAN_DIM: 3,
                    TRACER_DIM: len(utils.tracer_variables),
                },
                layout=self.layout,
            )
        return self._sizer

    @property
    def quantity_factory(self):
        if self._quantity_factory is None:
            self._quantity_factory = pace.util.QuantityFactory.from_backend(
                self.sizer, backend=self.backend
            )
        return self._quantity_factory

    def make_quantity(
        self,
        array,
        dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
        units="Unknown",
        origin=None,
        extent=None,
    ):
        if origin is None:
            origin = self.compute_origin()
        if extent is None:
            extent = self.domain_shape_compute()
        return pace.util.Quantity(
            array, dims=dims, units=units, origin=origin, extent=extent
        )

    def quantity_dict_update(
        self,
        data_dict,
        varname,
        dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
        units="Unknown",
    ):
        data_dict[varname + "_quantity"] = self.quantity_wrap(
            data_dict[varname], dims=dims, units=units
        )

    def quantity_wrap(
        self,
        data,
        dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
        units="Unknown",
    ):
        origin = self.sizer.get_origin(dims)
        extent = self.sizer.get_extent(dims)
        return pace.util.Quantity(
            data, dims=dims, units=units, origin=origin, extent=extent
        )

    def global_to_local_1d(self, global_value, subtile_index, subtile_length):
        return global_value - subtile_index * subtile_length

    def global_to_local_x(self, i_global):
        return self.global_to_local_1d(
            i_global, self.subtile_index[1], self.subtile_width_x
        )

    def global_to_local_y(self, j_global):
        return self.global_to_local_1d(
            j_global, self.subtile_index[0], self.subtile_width_y
        )

    def global_to_local_indices(self, i_global, j_global):
        i_local = self.global_to_local_x(i_global)
        j_local = self.global_to_local_y(j_global)
        return i_local, j_local

    def local_to_global_1d(self, local_value, subtile_index, subtile_length):
        return local_value + subtile_index * subtile_length

    def local_to_global_indices(self, i_local, j_local):
        i_global = self.local_to_global_1d(
            i_local, self.subtile_index[1], self.subtile_width_x
        )
        j_global = self.local_to_global_1d(
            j_local, self.subtile_index[0], self.subtile_width_y
        )
        return i_global, j_global

    def add_data(self, data_dict):
        self.data_fields.update(data_dict)
        for k, v in self.data_fields.items():
            setattr(self, k, v)

    def irange_compute(self):
        return range(self.is_, self.ie + 1)

    def irange_compute_x(self):
        return range(self.is_, self.ie + 2)

    def jrange_compute(self):
        return range(self.js, self.je + 1)

    def jrange_compute_y(self):
        return range(self.js, self.je + 2)

    def irange_domain(self):
        return range(self.isd, self.ied + 1)

    def jrange_domain(self):
        return range(self.jsd, self.jed + 1)

    def krange(self):
        return range(0, self.npz)

    def compute_interface(self):
        return self.slice_dict(self.compute_dict())

    def x3d_interface(self):
        return self.slice_dict(self.x3d_compute_dict())

    def y3d_interface(self):
        return self.slice_dict(self.y3d_compute_dict())

    def x3d_domain_interface(self):
        return self.slice_dict(self.x3d_domain_dict())

    def y3d_domain_interface(self):
        return self.slice_dict(self.y3d_domain_dict())

    def add_one(self, num):
        if num is None:
            return None
        return num + 1

    def slice_dict(self, d, ndim: int = 3):
        iters: str = "ijk" if ndim > 1 else "k"
        return tuple(
            [
                slice(d[f"{iters[i]}start"], self.add_one(d[f"{iters[i]}end"]))
                for i in range(ndim)
            ]
        )

    def default_domain_dict(self):
        return {
            "istart": self.isd,
            "iend": self.ied,
            "jstart": self.jsd,
            "jend": self.jed,
            "kstart": 0,
            "kend": self.npz - 1,
        }

    def default_dict_buffer_2d(self):
        mydict = self.default_domain_dict()
        mydict["iend"] += 1
        mydict["jend"] += 1
        return mydict

    def compute_dict(self):
        return {
            "istart": self.is_,
            "iend": self.ie,
            "jstart": self.js,
            "jend": self.je,
            "kstart": 0,
            "kend": self.npz - 1,
        }

    def compute_dict_buffer_2d(self):
        mydict = self.compute_dict()
        mydict["iend"] += 1
        mydict["jend"] += 1
        return mydict

    def default_buffer_k_dict(self):
        mydict = self.default_domain_dict()
        mydict["kend"] = self.npz
        return mydict

    def compute_buffer_k_dict(self):
        mydict = self.compute_dict()
        mydict["kend"] = self.npz
        return mydict

    def x3d_domain_dict(self):
        horizontal_dict = {
            "istart": self.isd,
            "iend": self.ied + 1,
            "jstart": self.jsd,
            "jend": self.jed,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def y3d_domain_dict(self):
        horizontal_dict = {
            "istart": self.isd,
            "iend": self.ied,
            "jstart": self.jsd,
            "jend": self.jed + 1,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def x3d_compute_dict(self):
        horizontal_dict = {
            "istart": self.is_,
            "iend": self.ie + 1,
            "jstart": self.js,
            "jend": self.je,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def y3d_compute_dict(self):
        horizontal_dict = {
            "istart": self.is_,
            "iend": self.ie,
            "jstart": self.js,
            "jend": self.je + 1,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def x3d_compute_domain_y_dict(self):
        horizontal_dict = {
            "istart": self.is_,
            "iend": self.ie + 1,
            "jstart": self.jsd,
            "jend": self.jed,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def y3d_compute_domain_x_dict(self):
        horizontal_dict = {
            "istart": self.isd,
            "iend": self.ied,
            "jstart": self.js,
            "jend": self.je + 1,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def domain_shape_full(self, *, add: Tuple[int, int, int] = (0, 0, 0)):
        """Domain shape for the full array including halo points."""
        return (self.nid + add[0], self.njd + add[1], self.npz + add[2])

    def domain_shape_compute(self, *, add: Tuple[int, int, int] = (0, 0, 0)):
        """Compute domain shape excluding halo points."""
        return (self.nic + add[0], self.njc + add[1], self.npz + add[2])

    def copy_right_edge(self, var, i_index, j_index):
        return np.copy(var[i_index:, :, :]), np.copy(var[:, j_index:, :])

    def insert_left_edge(self, var, edge_data_i, i_index, edge_data_j, j_index):
        if len(var.shape) < 3:
            var[:i_index, :] = edge_data_i
            var[:, :j_index] = edge_data_j
        else:
            var[:i_index, :, :] = edge_data_i
            var[:, :j_index, :] = edge_data_j

    def insert_right_edge(self, var, edge_data_i, i_index, edge_data_j, j_index):
        if len(var.shape) < 3:
            var[i_index:, :] = edge_data_i
            var[:, j_index:] = edge_data_j
        else:
            var[i_index:, :, :] = edge_data_i
            var[:, j_index:, :] = edge_data_j

    def uvar_edge_halo(self, var):
        return self.copy_right_edge(var, self.ie + 2, self.je + 1)

    def vvar_edge_halo(self, var):
        return self.copy_right_edge(var, self.ie + 1, self.je + 2)

    def compute_origin(self, add: Tuple[int, int, int] = (0, 0, 0)):
        """Start of the compute domain (e.g. (halo, halo, 0))"""
        return (self.is_ + add[0], self.js + add[1], add[2])

    def full_origin(self, add: Tuple[int, int, int] = (0, 0, 0)):
        """Start of the full array including halo points (e.g. (0, 0, 0))"""
        return (self.isd + add[0], self.jsd + add[1], add[2])

    def horizontal_starts_from_shape(self, shape):
        if shape[0:2] in [
            self.domain_shape_compute()[0:2],
            self.domain_shape_compute(add=(1, 0, 0))[0:2],
            self.domain_shape_compute(add=(0, 1, 0))[0:2],
            self.domain_shape_compute(add=(1, 1, 0))[0:2],
        ]:
            return self.is_, self.js
        elif shape[0:2] == (self.nic + 2, self.njc + 2):
            return self.is_ - 1, self.js - 1
        else:
            return 0, 0

    def get_halo_update_spec(
        self,
        shape,
        origin,
        halo_points,
        dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
    ) -> QuantityHaloSpec:
        """Build memory specifications for the halo update."""
        return self.grid_indexing.get_quantity_halo_spec(
            shape,
            origin,
            dims=dims,
            n_halo=halo_points,
            backend=self.backend,
        )

    @property
    def grid_indexing(self) -> "GridIndexing":
        return GridIndexing(
            domain=self.domain_shape_compute(),
            n_halo=self.halo,
            south_edge=self.south_edge,
            north_edge=self.north_edge,
            west_edge=self.west_edge,
            east_edge=self.east_edge,
        )

    @property
    def damping_coefficients(self) -> "DampingCoefficients":
        if self._damping_coefficients is not None:
            return self._damping_coefficients
        self._damping_coefficients = DampingCoefficients(
            divg_u=self.divg_u,
            divg_v=self.divg_v,
            del6_u=self.del6_u,
            del6_v=self.del6_v,
            da_min=self.da_min,
            da_min_c=self.da_min_c,
        )
        return self._damping_coefficients

    def set_damping_coefficients(self, damping_coefficients: "DampingCoefficients"):
        self._damping_coefficients = damping_coefficients

    @property
    def grid_data(self) -> "GridData":
        if self._grid_data is not None:
            return self._grid_data
        horizontal = HorizontalGridData(
            lon=self.bgrid1,
            lat=self.bgrid2,
            lon_agrid=self.agrid1,
            lat_agrid=self.agrid2,
            area=self.area,
            area_64=self.area_64,
            rarea=self.rarea,
            rarea_c=self.rarea_c,
            dx=self.dx,
            dy=self.dy,
            dxc=self.dxc,
            dyc=self.dyc,
            dxa=self.dxa,
            dya=self.dya,
            rdx=self.rdx,
            rdy=self.rdy,
            rdxc=self.rdxc,
            rdyc=self.rdyc,
            rdxa=self.rdxa,
            rdya=self.rdya,
            a11=self.a11,
            a12=self.a12,
            a21=self.a21,
            a22=self.a22,
            edge_w=self.edge_w,
            edge_e=self.edge_e,
            edge_s=self.edge_s,
            edge_n=self.edge_n,
        )
        vertical = VerticalGridData(ptop=-1.0e7, ks=-1)
        contravariant = ContravariantGridData(
            self.cosa,
            self.cosa_u,
            self.cosa_v,
            self.cosa_s,
            self.sina_u,
            self.sina_v,
            self.rsina,
            self.rsin_u,
            self.rsin_v,
            self.rsin2,
        )
        angle = AngleGridData(
            self.sin_sg1,
            self.sin_sg2,
            self.sin_sg3,
            self.sin_sg4,
            self.cos_sg1,
            self.cos_sg2,
            self.cos_sg3,
            self.cos_sg4,
        )
        self._grid_data = GridData(
            horizontal_data=horizontal,
            vertical_data=vertical,
            contravariant_data=contravariant,
            angle_data=angle,
        )
        return self._grid_data

    def set_grid_data(self, grid_data: "GridData"):
        self._grid_data = grid_data

    def make_grid_data(self, npx, npy, npz, communicator, backend):
        metric_terms = MetricTerms.from_tile_sizing(
            npx=npx, npy=npy, npz=npz, communicator=communicator, backend=backend
        )
        self.set_grid_data(GridData.new_from_metric_terms(metric_terms))
        self.set_damping_coefficients(
            DampingCoefficients.new_from_metric_terms(metric_terms)
        )
