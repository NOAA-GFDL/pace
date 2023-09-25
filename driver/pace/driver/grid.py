import abc
import dataclasses
from typing import ClassVar, Optional, Tuple

import f90nml

import pace.driver
import pace.dsl
import pace.physics
import pace.stencils
import pace.util.grid
from pace.stencils.testing import TranslateGrid
from pace.util import Communicator, QuantityFactory
from pace.util.grid import (
    DampingCoefficients,
    DriverGridData,
    GridData,
    MetricTerms,
    direct_transform,
)
from pace.util.grid.helper import (
    AngleGridData,
    ContravariantGridData,
    HorizontalGridData,
    VerticalGridData,
)
from pace.util.logging import pace_log
from pace.util.namelist import Namelist

from .registry import Registry


class GridInitializer(abc.ABC):
    @abc.abstractmethod
    def get_grid(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.Communicator,
    ) -> Tuple[DampingCoefficients, DriverGridData, GridData]:
        ...


@dataclasses.dataclass
class GridInitializerSelector(GridInitializer):
    """
    Dataclass for selecting the implementation of GridInitializer to use.

    Used to circumvent the issue that dacite expects static class definitions,
    but we would like to dynamically define which GridInitializer to use. Does this
    by representing the part of the yaml specification that asks which initializer
    to use, but deferring to the implementation in that initializer when called.
    """

    type: str
    config: GridInitializer
    registry: ClassVar[Registry] = Registry()

    @classmethod
    def register(cls, type_name):
        return cls.registry.register(type_name)

    def get_grid(
        self,
        quantity_factory: QuantityFactory,
        communicator: Communicator,
    ) -> Tuple[DampingCoefficients, DriverGridData, GridData]:
        return self.config.get_grid(
            quantity_factory=quantity_factory, communicator=communicator
        )

    @classmethod
    def from_dict(cls, config: dict):
        instance = cls.registry.from_dict(config)
        return cls(config=instance, type=config["type"])


@GridInitializerSelector.register("generated")
@dataclasses.dataclass
class GeneratedGridConfig(GridInitializer):
    """
    Configuration for a cubed-sphere grid computed from configuration.

    Attributes:
        stretch_factor: refinement amount
        lon_target: desired center longitude for refined tile (deg)
        lat_target: desired center latitude for refined tile (deg)
        restart_path: if given, load vertical grid from restart file
        grid_type: type of grid, 0 is a gnomonic cubed-sphere, 4 is doubly-periodic
        dx_const: constant x-width of grid cells on a dp-grid
        dy_const: constant y-width of grid cells on a dp-grid
        deglat: latitude to use for coriolis calculations on a dp-grid
    """

    stretch_factor: Optional[float] = 1.0
    lon_target: Optional[float] = 350.0
    lat_target: Optional[float] = -90.0
    restart_path: Optional[str] = None
    grid_type: Optional[int] = 0
    dx_const: Optional[float] = 1000.0
    dy_const: Optional[float] = 1000.0
    deglat: Optional[float] = 15.0

    def get_grid(
        self,
        quantity_factory: QuantityFactory,
        communicator: Communicator,
    ) -> Tuple[DampingCoefficients, DriverGridData, GridData]:
        metric_terms = MetricTerms(
            quantity_factory=quantity_factory,
            communicator=communicator,
            grid_type=self.grid_type,
            dx_const=self.dx_const,
            dy_const=self.dy_const,
            deglat=self.deglat,
        )
        if self.stretch_factor != 1:  # do horizontal grid transformation
            _transform_horizontal_grid(
                metric_terms, self.stretch_factor, self.lon_target, self.lat_target
            )

        horizontal_data = HorizontalGridData.new_from_metric_terms(metric_terms)
        if self.restart_path is not None:
            vertical_data = VerticalGridData.from_restart(
                self.restart_path, quantity_factory=quantity_factory
            )
        else:
            vertical_data = VerticalGridData.new_from_metric_terms(metric_terms)
        contravariant_data = ContravariantGridData.new_from_metric_terms(metric_terms)
        angle_data = AngleGridData.new_from_metric_terms(metric_terms)
        grid_data = GridData(
            horizontal_data=horizontal_data,
            vertical_data=vertical_data,
            contravariant_data=contravariant_data,
            angle_data=angle_data,
        )

        damping_coefficients = DampingCoefficients.new_from_metric_terms(metric_terms)
        driver_grid_data = DriverGridData.new_from_metric_terms(metric_terms)

        return damping_coefficients, driver_grid_data, grid_data


@GridInitializerSelector.register("serialbox")
@dataclasses.dataclass
class SerialboxGridConfig(GridInitializer):
    """
    Configuration for grid initialized from Serialbox data.
    """

    path: str

    @property
    def _f90_namelist(self) -> f90nml.Namelist:
        return f90nml.read(self.path + "/input.nml")

    @property
    def _namelist(self) -> Namelist:
        return Namelist.from_f90nml(self._f90_namelist)

    def _serializer(self, communicator: pace.util.Communicator):
        import serialbox

        serializer = serialbox.Serializer(
            serialbox.OpenModeKind.Read,
            self.path,
            "Generator_rank" + str(communicator.rank),
        )
        return serializer

    def _get_serialized_grid(
        self,
        communicator: pace.util.Communicator,
        backend: str,
    ) -> pace.stencils.testing.grid.Grid:  # type: ignore
        ser = self._serializer(communicator)
        grid = TranslateGrid.new_from_serialized_data(
            ser, communicator.rank, self._namelist.layout, backend
        ).python_grid()
        return grid

    def get_grid(
        self,
        quantity_factory: QuantityFactory,
        communicator: Communicator,
    ) -> Tuple[DampingCoefficients, DriverGridData, GridData]:
        backend = quantity_factory.zeros(
            dims=[pace.util.X_DIM, pace.util.Y_DIM], units="unknown"
        ).gt4py_backend

        pace_log.info("Using serialized grid data")
        grid = self._get_serialized_grid(communicator, backend)
        grid_data = grid.grid_data
        driver_grid_data = grid.driver_grid_data
        damping_coefficients = grid.damping_coefficients

        return damping_coefficients, driver_grid_data, grid_data


def _transform_horizontal_grid(
    metric_terms: MetricTerms,
    stretch_factor: float,
    lon_target: float,
    lat_target: float,
):
    """
    Uses the Schmidt transform to locally refine the horizontal grid.

    Args:
        metric_terms
        stretch_factor: refinement factor for tile 6
        lon_target: in degrees, lon of the new center for refined tile 6
        lat_target: in degrees, lat of the new center for refined tile 6

    Returns:
        updated metric terms
    """
    grid = metric_terms.grid
    lon_transform, lat_transform = direct_transform(
        lon=grid.data[:, :, 0],
        lat=grid.data[:, :, 1],
        stretch_factor=stretch_factor,
        lon_target=lon_target,
        lat_target=lat_target,
        np=grid.np,
    )
    grid.data[:, :, 0] = lon_transform[:]
    grid.data[:, :, 1] = lat_transform[:]

    metric_terms._grid.data[:] = grid.data[:]  # type: ignore[attr-defined]
    metric_terms._init_agrid()
