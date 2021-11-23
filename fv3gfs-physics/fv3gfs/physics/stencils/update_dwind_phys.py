import copy

import numpy as np
from gt4py import storage
from gt4py.gtscript import (
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    horizontal,
    interval,
)
from gt4py.storage.utils import idx_from_order

import fv3gfs.util
import fv3gfs.util.pace.gt4py_utils as utils
from fv3gfs.physics.global_constants import *
from fv3gfs.util.pace.typing import (
    Float,
    FloatField,
    FloatFieldI,
    FloatFieldIJ,
    Int,
    IntField,
)
from fv3gfs.util.quantity import Quantity
from fv3gfs.util.stencil import StencilFactory


def update_dwind_prep_stencil(
    u_dt: FloatField,
    v_dt: FloatField,
    vlon1: FloatFieldIJ,
    vlon2: FloatFieldIJ,
    vlon3: FloatFieldIJ,
    vlat1: FloatFieldIJ,
    vlat2: FloatFieldIJ,
    vlat3: FloatFieldIJ,
    ue_1: FloatField,
    ue_2: FloatField,
    ue_3: FloatField,
    ve_1: FloatField,
    ve_2: FloatField,
    ve_3: FloatField,
):
    with computation(PARALLEL), interval(...):
        v3_1 = u_dt * vlon1 + v_dt * vlat1
        v3_2 = u_dt * vlon2 + v_dt * vlat2
        v3_3 = u_dt * vlon3 + v_dt * vlat3
        ue_1 = v3_1[0, -1, 0] + v3_1
        ue_2 = v3_2[0, -1, 0] + v3_2
        ue_3 = v3_3[0, -1, 0] + v3_3
        ve_1 = v3_1[-1, 0, 0] + v3_1
        ve_2 = v3_2[-1, 0, 0] + v3_2
        ve_3 = v3_3[-1, 0, 0] + v3_3


def update_dwind_y_edge_south_stencil(
    ve_1: FloatField,
    ve_2: FloatField,
    ve_3: FloatField,
    vt_1: FloatField,
    vt_2: FloatField,
    vt_3: FloatField,
    edge_vect: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        vt_1 = edge_vect * ve_1[0, 1, 0] + (1.0 - edge_vect) * ve_1
        vt_2 = edge_vect * ve_2[0, 1, 0] + (1.0 - edge_vect) * ve_2
        vt_3 = edge_vect * ve_3[0, 1, 0] + (1.0 - edge_vect) * ve_3


def update_dwind_y_edge_north_stencil(
    ve_1: FloatField,
    ve_2: FloatField,
    ve_3: FloatField,
    vt_1: FloatField,
    vt_2: FloatField,
    vt_3: FloatField,
    edge_vect: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        vt_1 = edge_vect * ve_1[0, -1, 0] + (1.0 - edge_vect) * ve_1
        vt_2 = edge_vect * ve_2[0, -1, 0] + (1.0 - edge_vect) * ve_2
        vt_3 = edge_vect * ve_3[0, -1, 0] + (1.0 - edge_vect) * ve_3


def update_dwind_x_edge_west_stencil(
    ue_1: FloatField,
    ue_2: FloatField,
    ue_3: FloatField,
    ut_1: FloatField,
    ut_2: FloatField,
    ut_3: FloatField,
    edge_vect: FloatFieldI,
):
    with computation(PARALLEL), interval(...):
        ut_1 = edge_vect * ue_1[1, 0, 0] + (1.0 - edge_vect) * ue_1
        ut_2 = edge_vect * ue_2[1, 0, 0] + (1.0 - edge_vect) * ue_2
        ut_3 = edge_vect * ue_3[1, 0, 0] + (1.0 - edge_vect) * ue_3


def update_dwind_x_edge_east_stencil(
    ue_1: FloatField,
    ue_2: FloatField,
    ue_3: FloatField,
    ut_1: FloatField,
    ut_2: FloatField,
    ut_3: FloatField,
    edge_vect: FloatFieldI,
):
    with computation(PARALLEL), interval(...):
        ut_1 = edge_vect * ue_1[-1, 0, 0] + (1.0 - edge_vect) * ue_1
        ut_2 = edge_vect * ue_2[-1, 0, 0] + (1.0 - edge_vect) * ue_2
        ut_3 = edge_vect * ue_3[-1, 0, 0] + (1.0 - edge_vect) * ue_3


def copy3_stencil(
    in_field1: FloatField,
    in_field2: FloatField,
    in_field3: FloatField,
    out_field1: FloatField,
    out_field2: FloatField,
    out_field3: FloatField,
):
    with computation(PARALLEL), interval(...):
        out_field1 = in_field1
        out_field2 = in_field2
        out_field3 = in_field3


def update_uwind_stencil(
    u: FloatField,
    es1_1: FloatFieldIJ,
    es2_1: FloatFieldIJ,
    es3_1: FloatFieldIJ,
    ue_1: FloatField,
    ue_2: FloatField,
    ue_3: FloatField,
    dt5: float,
):
    with computation(PARALLEL), interval(...):
        # is: ie; js:je+1
        u = u + dt5 * (ue_1 * es1_1 + ue_2 * es2_1 + ue_3 * es3_1)


def update_vwind_stencil(
    v: FloatField,
    ew1_2: FloatFieldIJ,
    ew2_2: FloatFieldIJ,
    ew3_2: FloatFieldIJ,
    ve_1: FloatField,
    ve_2: FloatField,
    ve_3: FloatField,
    dt5: float,
):
    with computation(PARALLEL), interval(...):
        # is: ie+1; js:je
        v = v + dt5 * (ve_1 * ew1_2 + ve_2 * ew2_2 + ve_3 * ew3_2)


class AGrid2DGridPhysics:
    """
    Fortran name is update_dwinds_phys
    """

    def __init__(self, stencil_factory: StencilFactory, grid, namelist, grid_info):
        self.grid = grid
        self.namelist = namelist
        self._dt5 = 0.5 * self.namelist.dt_atmos
        self._im2 = int((self.grid.npx - 1) / 2) + 2
        self._jm2 = int((self.grid.npy - 1) / 2) + 2
        shape = self.grid.grid_indexing.max_shape
        self._ue_1 = utils.make_storage_from_shape(shape, init=True)
        self._ue_2 = utils.make_storage_from_shape(shape, init=True)
        self._ue_3 = utils.make_storage_from_shape(shape, init=True)
        self._ut_1 = utils.make_storage_from_shape(shape, init=True)
        self._ut_2 = utils.make_storage_from_shape(shape, init=True)
        self._ut_3 = utils.make_storage_from_shape(shape, init=True)
        self._ve_1 = utils.make_storage_from_shape(shape, init=True)
        self._ve_2 = utils.make_storage_from_shape(shape, init=True)
        self._ve_3 = utils.make_storage_from_shape(shape, init=True)
        self._vt_1 = utils.make_storage_from_shape(shape, init=True)
        self._vt_2 = utils.make_storage_from_shape(shape, init=True)
        self._vt_3 = utils.make_storage_from_shape(shape, init=True)
        self._update_dwind_prep_stencil = stencil_factory.from_origin_domain(
            update_dwind_prep_stencil,
            origin=(self.grid.halo - 1, self.grid.halo - 1, 0),
            domain=(self.grid.nic + 2, self.grid.njc + 2, self.grid.npz),
        )
        if self.grid.west_edge:
            je_lower = self.grid.global_to_local_y(min(self._jm2, self.grid.global_je))
            origin_lower = (self.grid.halo, self.grid.halo, 0)
            self._domain_lower_west = (
                1,
                je_lower - self.grid.js + 1,
                self.grid.npz,
            )
            if self.grid.global_js <= self._jm2:
                if self._domain_lower_west[1] > 0:
                    self._update_dwind_y_edge_south_stencil1 = (
                        stencil_factory.from_origin_domain(
                            update_dwind_y_edge_south_stencil,
                            origin=origin_lower,
                            domain=self._domain_lower_west,
                        )
                    )
            if self.grid.global_je > self._jm2:
                js_upper = self.grid.global_to_local_y(
                    max(self._jm2 + 1, self.grid.global_js)
                )
                origin_upper = (self.grid.halo, js_upper, 0)
                self._domain_upper_west = (
                    1,
                    self.grid.je - js_upper + 1,
                    self.grid.npz,
                )
                if self._domain_upper_west[1] > 0:
                    self._update_dwind_y_edge_north_stencil1 = (
                        stencil_factory.from_origin_domain(
                            update_dwind_y_edge_north_stencil,
                            origin=origin_upper,
                            domain=self._domain_upper_west,
                        )
                    )
                    self._copy3_stencil1 = stencil_factory.from_origin_domain(
                        copy3_stencil,
                        origin=origin_upper,
                        domain=self._domain_upper_west,
                    )
            if self.grid.global_js <= self._jm2 and self._domain_lower_west[1] > 0:
                self._copy3_stencil2 = stencil_factory.from_origin_domain(
                    copy3_stencil, origin=origin_lower, domain=self._domain_lower_west
                )
        if self.grid.east_edge:
            i_origin = shape[0] - self.grid.halo - 1
            je_lower = self.grid.global_to_local_y(min(self._jm2, self.grid.global_je))
            origin_lower = (i_origin, self.grid.halo, 0)
            self._domain_lower_east = (
                1,
                je_lower - self.grid.js + 1,
                self.grid.npz,
            )
            if self.grid.global_js <= self._jm2:
                if self._domain_lower_east[1] > 0:
                    self._update_dwind_y_edge_south_stencil2 = (
                        stencil_factory.from_origin_domain(
                            update_dwind_y_edge_south_stencil,
                            origin=origin_lower,
                            domain=self._domain_lower_east,
                        )
                    )

            if self.grid.global_je > self._jm2:
                js_upper = self.grid.global_to_local_y(
                    max(self._jm2 + 1, self.grid.global_js)
                )
                origin_upper = (i_origin, js_upper, 0)
                self._domain_upper_east = (
                    1,
                    self.grid.je - js_upper + 1,
                    self.grid.npz,
                )
                if self._domain_upper_east[1] > 0:
                    self._update_dwind_y_edge_north_stencil2 = (
                        stencil_factory.from_origin_domain(
                            update_dwind_y_edge_north_stencil,
                            origin=origin_upper,
                            domain=self._domain_upper_east,
                        )
                    )
                    self._copy3_stencil3 = stencil_factory.from_origin_domain(
                        copy3_stencil,
                        origin=origin_upper,
                        domain=self._domain_upper_east,
                    )
            if self.grid.global_js <= self._jm2 and self._domain_lower_east[1] > 0:
                self._copy3_stencil4 = stencil_factory.from_origin_domain(
                    copy3_stencil, origin=origin_lower, domain=self._domain_lower_east
                )
        if self.grid.south_edge:
            ie_lower = self.grid.global_to_local_x(min(self._im2, self.grid.global_ie))
            origin_lower = (self.grid.halo, self.grid.halo, 0)
            self._domain_lower_south = (
                ie_lower - self.grid.is_ + 1,
                1,
                self.grid.npz,
            )
            if self.grid.global_is <= self._im2:
                if self._domain_lower_south[0] > 0:
                    self._update_dwind_x_edge_west_stencil1 = (
                        stencil_factory.from_origin_domain(
                            update_dwind_x_edge_west_stencil,
                            origin=origin_lower,
                            domain=self._domain_lower_south,
                        )
                    )
            if self.grid.global_ie > self._im2:
                is_upper = self.grid.global_to_local_x(
                    max(self._im2 + 1, self.grid.global_is)
                )
                origin_upper = (is_upper, self.grid.halo, 0)
                self._domain_upper_south = (
                    self.grid.ie - is_upper + 1,
                    1,
                    self.grid.npz,
                )
                self._update_dwind_x_edge_east_stencil1 = (
                    stencil_factory.from_origin_domain(
                        update_dwind_x_edge_east_stencil,
                        origin=origin_upper,
                        domain=self._domain_upper_south,
                    )
                )
                self._copy3_stencil5 = stencil_factory.from_origin_domain(
                    copy3_stencil, origin=origin_upper, domain=self._domain_upper_south
                )
            if self.grid.global_is <= self._im2 and self._domain_lower_south[0] > 0:
                self._copy3_stencil6 = stencil_factory.from_origin_domain(
                    copy3_stencil, origin=origin_lower, domain=self._domain_lower_south
                )
        if self.grid.north_edge:
            j_origin = shape[1] - self.grid.halo - 1
            ie_lower = self.grid.global_to_local_x(min(self._im2, self.grid.global_ie))
            origin_lower = (self.grid.halo, j_origin, 0)
            self._domain_lower_north = (
                ie_lower - self.grid.is_ + 1,
                1,
                self.grid.npz,
            )
            if self.grid.global_is < self._im2:
                if self._domain_lower_north[0] > 0:
                    self._update_dwind_x_edge_west_stencil2 = (
                        stencil_factory.from_origin_domain(
                            update_dwind_x_edge_west_stencil,
                            origin=origin_lower,
                            domain=self._domain_lower_north,
                        )
                    )
            if self.grid.global_ie >= self._im2:
                is_upper = self.grid.global_to_local_x(
                    max(self._im2 + 1, self.grid.global_is)
                )
                origin_upper = (is_upper, j_origin, 0)
                self._domain_upper_north = (
                    self.grid.ie - is_upper + 1,
                    1,
                    self.grid.npz,
                )
                if self._domain_upper_north[0] > 0:
                    self._update_dwind_x_edge_east_stencil2 = (
                        stencil_factory.from_origin_domain(
                            update_dwind_x_edge_east_stencil,
                            origin=origin_upper,
                            domain=self._domain_upper_north,
                        )
                    )
                    self._copy3_stencil7 = stencil_factory.from_origin_domain(
                        copy3_stencil,
                        origin=origin_upper,
                        domain=self._domain_upper_north,
                    )
            if self.grid.global_is < self._im2 and self._domain_lower_north[0] > 0:
                self._copy3_stencil8 = stencil_factory.from_origin_domain(
                    copy3_stencil, origin=origin_lower, domain=self._domain_lower_north
                )
        self._update_uwind_stencil = stencil_factory.from_origin_domain(
            update_uwind_stencil,
            origin=(self.grid.halo, self.grid.halo, 0),
            domain=(self.grid.nic, self.grid.njc + 1, self.grid.npz),
        )
        self._update_vwind_stencil = stencil_factory.from_origin_domain(
            update_vwind_stencil,
            origin=(self.grid.halo, self.grid.halo, 0),
            domain=(self.grid.nic + 1, self.grid.njc, self.grid.npz),
        )
        # [TODO] The following is waiting on grid code vlat and vlon
        self._vlon1 = grid_info["vlon1"]
        self._vlon2 = grid_info["vlon2"]
        self._vlon3 = grid_info["vlon3"]
        self._vlat1 = grid_info["vlat1"]
        self._vlat2 = grid_info["vlat2"]
        self._vlat3 = grid_info["vlat3"]
        self._edge_vect_w = grid_info["edge_vect_w"]
        self._edge_vect_e = grid_info["edge_vect_e"]
        self._edge_vect_s = grid_info["edge_vect_s"]
        self._edge_vect_n = grid_info["edge_vect_n"]
        self._es1_1 = grid_info["es1_1"]
        self._es2_1 = grid_info["es2_1"]
        self._es3_1 = grid_info["es3_1"]
        self._ew1_2 = grid_info["ew1_2"]
        self._ew2_2 = grid_info["ew2_2"]
        self._ew3_2 = grid_info["ew3_2"]

    def __call__(
        self,
        u: FloatField,
        v: FloatField,
        u_dt: FloatField,
        v_dt: FloatField,
    ):
        """
        Transforms the wind tendencies from A grid to D grid for the final update
        """
        self._update_dwind_prep_stencil(
            u_dt,
            v_dt,
            self._vlon1,
            self._vlon2,
            self._vlon3,
            self._vlat1,
            self._vlat2,
            self._vlat3,
            self._ue_1,
            self._ue_2,
            self._ue_3,
            self._ve_1,
            self._ve_2,
            self._ve_3,
        )
        if self.grid.west_edge:
            if self.grid.global_js <= self._jm2:
                if self._domain_lower_west[1] > 0:
                    self._update_dwind_y_edge_south_stencil1(
                        self._ve_1,
                        self._ve_2,
                        self._ve_3,
                        self._vt_1,
                        self._vt_2,
                        self._vt_3,
                        self._edge_vect_w,
                    )
            if self.grid.global_je > self._jm2:
                if self._domain_upper_west[1] > 0:
                    self._update_dwind_y_edge_north_stencil1(
                        self._ve_1,
                        self._ve_2,
                        self._ve_3,
                        self._vt_1,
                        self._vt_2,
                        self._vt_3,
                        self._edge_vect_w,
                    )
                    self._copy3_stencil1(
                        self._vt_1,
                        self._vt_2,
                        self._vt_3,
                        self._ve_1,
                        self._ve_2,
                        self._ve_3,
                    )
            if self.grid.global_js <= self._jm2 and self._domain_lower_west[1] > 0:
                self._copy3_stencil2(
                    self._vt_1,
                    self._vt_2,
                    self._vt_3,
                    self._ve_1,
                    self._ve_2,
                    self._ve_3,
                )
        if self.grid.east_edge:
            if self.grid.global_js <= self._jm2:
                if self._domain_lower_east[1] > 0:
                    self._update_dwind_y_edge_south_stencil2(
                        self._ve_1,
                        self._ve_2,
                        self._ve_3,
                        self._vt_1,
                        self._vt_2,
                        self._vt_3,
                        self._edge_vect_e,
                    )
            if self.grid.global_je > self._jm2:
                if self._domain_upper_east[1] > 0:
                    self._update_dwind_y_edge_north_stencil2(
                        self._ve_1,
                        self._ve_2,
                        self._ve_3,
                        self._vt_1,
                        self._vt_2,
                        self._vt_3,
                        self._edge_vect_e,
                    )
                    self._copy3_stencil3(
                        self._vt_1,
                        self._vt_2,
                        self._vt_3,
                        self._ve_1,
                        self._ve_2,
                        self._ve_3,
                    )
            if self.grid.global_js <= self._jm2 and self._domain_lower_east[1] > 0:
                self._copy3_stencil4(
                    self._vt_1,
                    self._vt_2,
                    self._vt_3,
                    self._ve_1,
                    self._ve_2,
                    self._ve_3,
                )
        if self.grid.south_edge:
            if self.grid.global_is <= self._im2:
                if self._domain_lower_south[0] > 0:
                    self._update_dwind_x_edge_west_stencil1(
                        self._ue_1,
                        self._ue_2,
                        self._ue_3,
                        self._ut_1,
                        self._ut_2,
                        self._ut_3,
                        self._edge_vect_s,
                    )
            if self.grid.global_ie > self._im2:
                if self._domain_upper_south:
                    self._update_dwind_x_edge_east_stencil1(
                        self._ue_1,
                        self._ue_2,
                        self._ue_3,
                        self._ut_1,
                        self._ut_2,
                        self._ut_3,
                        self._edge_vect_s,
                    )
                    self._copy3_stencil5(
                        self._ut_1,
                        self._ut_2,
                        self._ut_3,
                        self._ue_1,
                        self._ue_2,
                        self._ue_3,
                    )
            if self.grid.global_is <= self._im2 and self._domain_lower_south[0] > 0:
                self._copy3_stencil6(
                    self._ut_1,
                    self._ut_2,
                    self._ut_3,
                    self._ue_1,
                    self._ue_2,
                    self._ue_3,
                )
        if self.grid.north_edge:
            if self.grid.global_is < self._im2:
                if self._domain_lower_north[0] > 0:
                    self._update_dwind_x_edge_west_stencil2(
                        self._ue_1,
                        self._ue_2,
                        self._ue_3,
                        self._ut_1,
                        self._ut_2,
                        self._ut_3,
                        self._edge_vect_n,
                    )
            if self.grid.global_ie >= self._im2:
                if self._domain_upper_north[0] > 0:
                    self._update_dwind_x_edge_east_stencil2(
                        self._ue_1,
                        self._ue_2,
                        self._ue_3,
                        self._ut_1,
                        self._ut_2,
                        self._ut_3,
                        self._edge_vect_n,
                    )
                    self._copy3_stencil7(
                        self._ut_1,
                        self._ut_2,
                        self._ut_3,
                        self._ue_1,
                        self._ue_2,
                        self._ue_3,
                    )
            if self.grid.global_is < self._im2 and self._domain_lower_north[0] > 0:
                self._copy3_stencil8(
                    self._ut_1,
                    self._ut_2,
                    self._ut_3,
                    self._ue_1,
                    self._ue_2,
                    self._ue_3,
                )
        self._update_uwind_stencil(
            u,
            self._es1_1,
            self._es2_1,
            self._es3_1,
            self._ue_1,
            self._ue_2,
            self._ue_3,
            self._dt5,
        )
        self._update_vwind_stencil(
            v,
            self._ew1_2,
            self._ew2_2,
            self._ew3_2,
            self._ve_1,
            self._ve_2,
            self._ve_3,
            self._dt5,
        )
