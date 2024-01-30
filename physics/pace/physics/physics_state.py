from dataclasses import InitVar, dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional

import xarray as xr

import ndsl.dsl.gt4py_utils as gt_utils
import ndsl.util
from ndsl.dsl.typing import Float
from pace.physics.stencils.microphysics import MicrophysicsState

from ._config import PHYSICS_PACKAGES


@dataclass()
class PhysicsState:
    qvapor: ndsl.util.Quantity = field(
        metadata={
            "name": "specific_humidity",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "kg/kg",
        }
    )
    qliquid: ndsl.util.Quantity = field(
        metadata={
            "name": "cloud_water_mixing_ratio",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qice: ndsl.util.Quantity = field(
        metadata={
            "name": "cloud_ice_mixing_ratio",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qrain: ndsl.util.Quantity = field(
        metadata={
            "name": "rain_mixing_ratio",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qsnow: ndsl.util.Quantity = field(
        metadata={
            "name": "snow_mixing_ratio",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qgraupel: ndsl.util.Quantity = field(
        metadata={
            "name": "graupel_mixing_ratio",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qo3mr: ndsl.util.Quantity = field(
        metadata={
            "name": "ozone_mixing_ratio",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qsgs_tke: ndsl.util.Quantity = field(
        metadata={
            "name": "turbulent_kinetic_energy",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "m**2/s**2",
            "intent": "inout",
        }
    )
    qcld: ndsl.util.Quantity = field(
        metadata={
            "name": "cloud_fraction",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    pt: ndsl.util.Quantity = field(
        metadata={
            "name": "air_temperature",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "degK",
            "intent": "inout",
        }
    )
    delp: ndsl.util.Quantity = field(
        metadata={
            "name": "pressure_thickness_of_atmospheric_layer",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    delz: ndsl.util.Quantity = field(
        metadata={
            "name": "vertical_thickness_of_atmospheric_layer",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "m",
            "intent": "inout",
        }
    )
    ua: ndsl.util.Quantity = field(
        metadata={
            "name": "eastward_wind",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    va: ndsl.util.Quantity = field(
        metadata={
            "name": "northward_wind",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "m/s",
        }
    )
    w: ndsl.util.Quantity = field(
        metadata={
            "name": "vertical_wind",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    omga: ndsl.util.Quantity = field(
        metadata={
            "name": "vertical_pressure_velocity",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "Pa/s",
            "intent": "inout",
        }
    )
    physics_updated_specific_humidity: ndsl.util.Quantity = field(
        metadata={
            "name": "physics_updated_specific_humidity",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "kg/kg",
        }
    )
    physics_updated_qliquid: ndsl.util.Quantity = field(
        metadata={
            "name": "physics_updated_liquid_water_mixing_ratio",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    physics_updated_qice: ndsl.util.Quantity = field(
        metadata={
            "name": "physics_updated_ice_water_mixing_ratio",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    physics_updated_qrain: ndsl.util.Quantity = field(
        metadata={
            "name": "physics_updated_rain_water_mixing_ratio",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    physics_updated_qsnow: ndsl.util.Quantity = field(
        metadata={
            "name": "physics_updated_snow_mixing_ratio",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    physics_updated_qgraupel: ndsl.util.Quantity = field(
        metadata={
            "name": "physics_updated_graupel_mixing_ratio",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    physics_updated_cloud_fraction: ndsl.util.Quantity = field(
        metadata={
            "name": "physics_cloud_fraction",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    physics_updated_pt: ndsl.util.Quantity = field(
        metadata={
            "name": "physics_air_temperature",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "degK",
            "intent": "inout",
        }
    )
    physics_updated_ua: ndsl.util.Quantity = field(
        metadata={
            "name": "physics_eastward_wind",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    physics_updated_va: ndsl.util.Quantity = field(
        metadata={
            "name": "physics_northward_wind",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    delprsi: ndsl.util.Quantity = field(
        metadata={
            "name": "model_level_pressure_thickness_in_physics",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    phii: ndsl.util.Quantity = field(
        metadata={
            "name": "interface_geopotential_height",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_INTERFACE_DIM],
            "units": "m",
            "intent": "inout",
        }
    )
    phil: ndsl.util.Quantity = field(
        metadata={
            "name": "layer_geopotential_height",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "m",
            "intent": "inout",
        }
    )
    dz: ndsl.util.Quantity = field(
        metadata={
            "name": "geopotential_height_thickness",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "m",
            "intent": "inout",
        }
    )
    wmp: ndsl.util.Quantity = field(
        metadata={
            "name": "layer_mean_vertical_velocity_microph",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    prsi: ndsl.util.Quantity = field(
        metadata={
            "name": "interface_pressure",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_INTERFACE_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    prsik: ndsl.util.Quantity = field(
        metadata={
            "name": "log_interface_pressure",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_INTERFACE_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    land: ndsl.util.Quantity = field(
        metadata={
            "name": "land_mask",
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_DIM],
            "units": "-",
            "intent": "in",
        }
    )
    quantity_factory: InitVar[ndsl.util.QuantityFactory]
    schemes: InitVar[List[PHYSICS_PACKAGES]]

    def __post_init__(
        self,
        quantity_factory: ndsl.util.QuantityFactory,
        schemes: List[PHYSICS_PACKAGES],
    ):
        # storage for tendency variables not in PhysicsState
        if "GFS_microphysics" in [scheme.value for scheme in schemes]:
            tendency = quantity_factory.zeros(
                [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
                "unknown",
                dtype=Float,
            )
            self.microphysics: Optional[MicrophysicsState] = MicrophysicsState(
                pt=self.pt,
                qvapor=self.qvapor,
                qliquid=self.qliquid,
                qrain=self.qrain,
                qice=self.qice,
                qsnow=self.qsnow,
                qgraupel=self.qgraupel,
                qcld=self.qcld,
                ua=self.ua,
                va=self.va,
                delp=self.delp,
                delz=self.delz,
                omga=self.omga,
                delprsi=self.delprsi,
                wmp=self.wmp,
                dz=self.dz,
                tendency=tendency,
                land=self.land,
            )
        else:
            self.microphysics = None

    @classmethod
    def init_zeros(
        cls, quantity_factory, schemes: List[PHYSICS_PACKAGES]
    ) -> "PhysicsState":
        initial_arrays = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                initial_arrays[_field.name] = quantity_factory.zeros(
                    _field.metadata["dims"],
                    _field.metadata["units"],
                    dtype=Float,
                ).data
        return cls(
            **initial_arrays,
            quantity_factory=quantity_factory,
            schemes=schemes,
        )

    @classmethod
    def init_from_storages(
        cls,
        storages: Mapping[str, Any],
        sizer: ndsl.util.GridSizer,
        quantity_factory: ndsl.util.QuantityFactory,
        schemes: List[PHYSICS_PACKAGES],
    ) -> "PhysicsState":
        inputs: Dict[str, ndsl.util.Quantity] = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                dims = _field.metadata["dims"]
                quantity = ndsl.util.Quantity(
                    storages[_field.name],
                    dims,
                    _field.metadata["units"],
                    origin=sizer.get_origin(dims),
                    extent=sizer.get_extent(dims),
                )
                inputs[_field.name] = quantity
        return cls(**inputs, quantity_factory=quantity_factory, schemes=schemes)

    @property
    def xr_dataset(self):
        data_vars = {}
        for name, field_info in self.__dataclass_fields__.items():
            if name not in ["quantity_factory", "schemes"]:
                if issubclass(field_info.type, ndsl.util.Quantity):
                    dims = [
                        f"{dim_name}_{name}" for dim_name in field_info.metadata["dims"]
                    ]
                    data_vars[name] = xr.DataArray(
                        gt_utils.asarray(getattr(self, name).data),
                        dims=dims,
                        attrs={
                            "long_name": field_info.metadata["name"],
                            "units": field_info.metadata.get("units", "unknown"),
                        },
                    )
        return xr.Dataset(data_vars=data_vars)
