import abc
import dataclasses
import warnings
from collections.abc import Mapping
from dataclasses import Field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ndsl import Quantity, DiagManagerMonitor
from ndsl.constants import K_DIM, K_INTERFACE_DIM, RGRAV
from ndsl.dsl.dace.orchestration import dace_inhibitor
from ndsl.dsl.typing import Float
from ndsl.grid import GridData
from ndsl.monitor import Monitor, ZarrMonitor
from ndsl.monitor.netcdf_monitor import NetCDFMonitor
from ndsl.typing import Communicator
from pace.state import DriverState
from pyfv3 import DycoreState


try:
    import zarr.storage as zarr_storage
except ModuleNotFoundError:
    zarr_storage = None


class Diagnostics(abc.ABC):
    @abc.abstractmethod
    def store(self, time: datetime | timedelta, state: DriverState): ...

    @abc.abstractmethod
    def store_grid(self, grid_data: GridData): ...

    @abc.abstractmethod
    def cleanup(self): ...


@dataclasses.dataclass
class ZSelect:
    level: int
    names: list[str]

    def select_data(self, state: DycoreState):
        output = {}
        for name in self.names:
            if name not in state.__dict__.keys():
                raise ValueError(f"Invalid state variable {name} for level select")
            assert len(getattr(state, name).dims) > 2
            if getattr(state, name).dims[2] != (K_DIM or K_INTERFACE_DIM):
                raise ValueError(
                    f"z_select only works for state variables with dimension (x, y, z). \
                        \n {name} has dimension {getattr(state, name).dims}"
                )
            var_name = f"{name}_z{self.level}"
            output[var_name] = Quantity(
                getattr(state, name).data[:, :, self.level],
                dims=getattr(state, name).dims[0:2],
                origin=getattr(state, name).origin[0:2],
                extent=getattr(state, name).extent[0:2],
                units=getattr(state, name).units,
                backend=getattr(state, name).backend,
            )
        return output


@dataclasses.dataclass(frozen=True)
class DiagnosticsConfig:
    """
    Attributes:
        path: directory to save diagnostics if given, otherwise no diagnostics
            will be stored
        output_format: one of "zarr" or "netcdf", be careful when using the "netcdf"
            format as this requires all diagnostics to be stored in memory on the
            root rank before saving, which can cause out-of-memory errors if the
            global data size or number of variables is too large
        time_chunk_size: number of timesteps stored in each netcdf file, only used if
            output_format is "netcdf"
        names: state variables to save as diagnostics
        derived_names: derived diagnostics to save
        z_select: save a vertical slice of a 3D state
    """

    path: str | None = None
    output_format: str = "zarr"
    time_chunk_size: int = 1
    names: list[str] = dataclasses.field(default_factory=list)
    derived_names: list[str] = dataclasses.field(default_factory=list)
    z_select: list[ZSelect] = dataclasses.field(default_factory=list)
    precision: str = "Float"

    def __post_init__(self):
        if (len(self.names) > 0 or len(self.derived_names) > 0) and \
            (self.path is None and self.output_format != "diag_manager"):
            raise ValueError(
                "DiagnosticsConfig.path must be given to enable diagnostics"
            )
        if self.output_format not in ["zarr", "netcdf", "diag_manager"]:
            raise ValueError(
                f"output_format must be one of 'zarr', 'netcdf', or 'diag_manager', got {self.output_format}"
            )
        if self.precision not in ["Float", "float32", "float64"]:
            raise ValueError(
                f"precision must be one of 'Float', 'float32', or 'float64got {self.precision}"
            )

    def diagnostics_factory(self, communicator: Communicator) -> Diagnostics:
        """
        Create a diagnostics object.

        Args:
            communicator: provides global communication e.g. to gather state
                or to coordinate filesystem access between ranks
        """
        if self.path is None and self.output_format != "diag_manager":
            return NullDiagnostics()

        if self.output_format != "diag_manager":
            if not Path(self.path).exists():
                Path(self.path).mkdir()

        if self.output_format == "zarr":
            store = zarr_storage.DirectoryStore(path=self.path)
            monitor: Monitor = ZarrMonitor(
                store=store,
                partitioner=communicator.partitioner,
                mpi_comm=communicator.comm,
            )
        elif self.output_format == "netcdf":
            if self.precision == "Float":
                precision = Float
            elif self.precision == "float32":
                precision = np.float32
            elif self.precision == "float64":
                precision = np.float64
            monitor = NetCDFMonitor(
                path=self.path,
                communicator=communicator,
                time_chunk_size=self.time_chunk_size,
                precision=precision,
            )
        elif self.output_format == "diag_manager":
            monitor = DiagManagerMonitor(
                domain_id=communicator.pyfms_domain_id,
            )
        else:
            raise ValueError(
                f"output_format must be one of 'zarr', 'netcdf', or 'diag_manager', got {self.output_format}"
            )

        return MonitorDiagnostics(
            monitor=monitor,
            names=self.names,
            derived_names=self.derived_names,
            z_select=self.z_select,
        )


class MonitorDiagnostics(Diagnostics):
    """Diagnostics that save to a sympl-style Monitor."""

    def __init__(
        self,
        monitor: Monitor,
        names: list[str],
        derived_names: list[str],
        z_select: list[ZSelect],
    ):
        """
        Args:
            monitor: a sympl-style Monitor object
            names: list of names of diagnostics to save
            derived_names: list of names of derived diagnostics to save
        """
        self.names = names
        self.derived_names = derived_names
        self.z_select = z_select
        self.monitor = monitor

    @dace_inhibitor
    def store(self, time: datetime | timedelta, state: DriverState):
        monitor_state = {"time": time}
        for name in self.names:
            try:
                quantity = getattr(state.dycore_state, name)
            except AttributeError:
                quantity = getattr(state.physics_state, name)
            monitor_state[name] = quantity
        derived_state = self._get_derived_state(state)
        level_select_state = self._get_z_select_state(state.dycore_state)
        monitor_state.update(derived_state)
        monitor_state.update(level_select_state)
        self.monitor.store(monitor_state)

    def _get_derived_state(self, state: DriverState):
        output = {}
        if len(self.derived_names) > 0:
            for name in self.derived_names:
                if name.startswith("column_integrated_"):
                    tracer = name[len("column_integrated_") :]
                    output[name] = _compute_column_integral(
                        name,
                        getattr(state.dycore_state, tracer),
                        state.dycore_state.delp,
                    )
                else:
                    warnings.warn(
                        f"{name} is not a supported diagnostic variable.", stacklevel=2
                    )
        return output

    def _get_z_select_state(self, state: DycoreState):
        z_select_state = {}
        for zselect in self.z_select:
            z_select_state.update(zselect.select_data(state))
        return z_select_state

    def store_grid(self, grid_data: GridData):
        zarr_grid = {
            "lat": grid_data.lat,
            "lon": grid_data.lon,
            "lon_agrid": grid_data.lon_agrid,
            "lat_agrid": grid_data.lat_agrid,
        }
        for k, v in zarr_grid.items():
            self.monitor.store_constant({k: v})

    def cleanup(self):
        self.monitor.cleanup()


class NullDiagnostics(Diagnostics):
    """Diagnostics that do nothing."""

    def store(self, time: datetime | timedelta, state: DriverState):
        pass

    def store_grid(self, grid_data: GridData):
        pass

    def cleanup(self):
        pass


def register_diag_manager_fields(
    *,
    dataclass_fields: Mapping[str, Field[Any]],
    monitor: Any,
    init_time: datetime,
    field_names: list[str],
    module_name: str,
    dtype: Any,
    use_metadata_name: bool = False,
) -> None:
    """Register selected dataclass fields with a diag_manager monitor.

    The input list is updated in place by removing names that are registered.
    """
    for field_name in list(field_names):
        dataclass_field = dataclass_fields.get(field_name)
        if dataclass_field is None:
            continue

        dims = dataclass_field.metadata.get("dims", "unknown")
        units = dataclass_field.metadata.get("units", "unknown")
        if use_metadata_name:
            diag_field_name = dataclass_field.metadata.get("name", field_name)
        else:
            diag_field_name = field_name

        monitor.register_field(
            module_name=module_name,
            field_name=diag_field_name,
            dims=dims,
            units=units,
            init_time=init_time,
            dtype=dtype,
        )
        field_names.remove(field_name)


def _compute_column_integral(name: str, q_in: Quantity, delp: Quantity) -> Quantity:
    """
    Compute column integrated mixing ratio (e.g., total liquid water path)

    Args:
        name: name of the tracer
        q_in: tracer mixing ratio
        delp: pressure thickness of atmospheric layer

    Returns:
        The column integral.
    """
    if len(q_in.dims) <= 2:
        raise RuntimeError("This function assumes that q_in is at least 3-dimensional.")

    if q_in.dims[2] != K_DIM:
        raise RuntimeError(
            "This function assumes the z-dimension is the third dimension"
        )

    k_slice = slice(q_in.origin[2], q_in.origin[2] + q_in.extent[2])
    return Quantity(
        RGRAV
        * q_in.np.sum(q_in.data[:, :, k_slice] * delp.data[:, :, k_slice], axis=2),
        dims=tuple(q_in.dims[:2]) + tuple(q_in.dims[3:]),
        origin=q_in.metadata.origin[0:2],
        extent=(q_in.metadata.extent[0], q_in.metadata.extent[1]),
        units="kg/m**2",
        backend=q_in.backend,
    )
