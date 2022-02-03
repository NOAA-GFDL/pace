import abc
import dataclasses
import functools
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union

import click
import dacite
import yaml
import zarr.storage
from mpi4py import MPI

import fv3core
import fv3core.initialization.baroclinic as baroclinic_init
import fv3gfs.physics
import pace.dsl
import pace.stencils
import pace.util
import pace.util.grid

# TODO: move update_atmos_state into pace.driver
from pace.stencils import update_atmos_state
from pace.util.grid import DampingCoefficients


@dataclasses.dataclass
class DriverState:
    dycore_state: fv3core.DycoreState
    physics_state: fv3gfs.physics.PhysicsState
    metric_terms: pace.util.grid.MetricTerms


class InitializationConfig(abc.ABC):
    @property
    @abc.abstractmethod
    def start_time(self) -> datetime:
        ...

    @abc.abstractmethod
    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
    ) -> DriverState:
        ...


@dataclasses.dataclass
class BaroclinicConfig(InitializationConfig):
    """
    Configuration for baroclinic initialization.
    """

    @property
    def start_time(self) -> datetime:
        # TODO: instead of arbitrary start time, enable use of timedeltas
        return datetime(2000, 1, 1)

    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
    ) -> DriverState:
        metric_terms = pace.util.grid.MetricTerms(
            quantity_factory=quantity_factory, communicator=communicator
        )
        dycore_state = baroclinic_init.init_baroclinic_state(
            metric_terms,
            adiabatic=False,
            hydrostatic=False,
            moist_phys=True,
            comm=communicator,
        )
        physics_state = fv3gfs.physics.PhysicsState.init_zeros(
            quantity_factory=quantity_factory, active_packages=["microphysics"]
        )
        return DriverState(
            dycore_state=dycore_state,
            physics_state=physics_state,
            metric_terms=metric_terms,
        )


@dataclasses.dataclass
class RestartConfig(InitializationConfig):
    """
    Configuration for restart initialization.
    """

    path: str

    @property
    def start_time(self) -> datetime:
        return datetime(2000, 1, 1)

    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
    ) -> DriverState:
        metric_terms = pace.util.grid.MetricTerms(
            quantity_factory=quantity_factory, communicator=communicator
        )
        state = pace.util.open_restart(
            dirname=self.path,
            communicator=communicator,
            quantity_factory=quantity_factory,
        )
        raise NotImplementedError()


@dataclasses.dataclass
class SerialboxConfig(InitializationConfig):
    """
    Configuration for Serialbox initialization.
    """

    path: str
    npx: int
    npy: int
    npz: int
    layout: List[int]
    backend: str

    @property
    def start_time(self) -> datetime:
        return datetime(2000, 1, 1)

    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
    ) -> DriverState:
        raise NotImplementedError()

    def _initialize_dycore_state(
        self, communicator: pace.util.CubedSphereCommunicator
    ) -> fv3core.DycoreState:
        # TODO: this code currently depends on namelist and stencil_factory,
        # update the config to require a namelist file or load the file from
        # the place we know it exists in
        # the file path, and use that to make a factory.
        #
        # import sys
        # sys.path.append("/usr/local/serialbox/python/")
        # import serialbox

        # serializer = serialbox.Serializer(
        #     serialbox.OpenModeKind.Read,
        #     self._data_dir,
        #     "Generator_rank" + str(self._comm.rank),
        # )

        # grid = fv3core._config.make_grid(
        #     npx=self.npx,
        #     npy=self.npy,
        #     npz=self.npz,
        #     layout=self.layout,
        #     rank=communicator.rank,
        #     backend=self.backend,
        # )
        # savepoint_in = serializer.get_savepoint("FVDynamics-In")[0]
        # translate_object = TranslateFVDynamics([grid], namelist, stencil_factory)
        # input_data = translate_object.collect_input_data(serializer, savepoint_in)
        # dycore_state = translate_object.state_from_inputs(input_data)
        # return dycore_state
        pass

    def _init_physics_state_from_dycore_state(
        self,
        dycore_state: fv3core.DycoreState,
        quantity_factory: pace.util.QuantityFactory,
    ) -> fv3gfs.physics.PhysicsState:
        initial_storages = {}
        dycore_fields = dataclasses.fields(fv3core.DycoreState)
        for field in dataclasses.fields(fv3gfs.physics.PhysicsState):
            metadata = field.metadata
            matches = [
                f
                for f in dycore_fields
                if field.name == f.name
                and metadata["name"] == f.metadata["name"]
                and metadata["units"] == f.metadata["units"]
            ]
            if len(matches) > 0:
                initial_storages[field.name] = getattr(dycore_state, field.name)
            else:
                initial_storages[field.name] = quantity_factory.zeros(
                    [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
                    field.metadata["units"],
                    dtype=float,
                )
        return fv3gfs.physics.PhysicsState(
            **initial_storages,
            quantity_factory=quantity_factory,
            active_packages=["microphysics"],
        )


@dataclasses.dataclass(frozen=True)
class DiagnosticsConfig:
    path: str
    names: List[str] = dataclasses.field(default_factory=list)


class Diagnostics:
    def __init__(
        self,
        config: DiagnosticsConfig,
        partitioner: pace.util.CubedSpherePartitioner,
        comm,
    ):
        self.config = config
        store = zarr.storage.DirectoryStore(path=self.config.path)
        self.monitor = pace.util.ZarrMonitor(
            store=store, partitioner=partitioner, mpi_comm=comm
        )

    def store(self, time: datetime, state: DriverState):
        zarr_state = {"time": time}
        for name in self.config.names:
            try:
                quantity = getattr(state.dycore_state, name)
            except AttributeError:
                quantity = getattr(state.physics_state, name)
            zarr_state[name] = quantity
        assert time is not None
        self.monitor.store(zarr_state)


@dataclasses.dataclass(frozen=True)
class DriverConfig:
    """
    Configuration for a run of the Pace model.

    Attributes:
        stencil_config: configuration for stencil compilation
        initialization_type: must be "baroclinic" or "restart"
        initialization_config: configuration for the chosen initialization
            type, see documentation for its corresponding configuration
            dataclass
        nx_tile: number of gridpoints along the horizontal dimension of a cube
            tile face, same value used for both horizontal dimensions
        nz: number of gridpoints in the vertical dimension
        layout: number of ranks along the x and y dimensions
        dt_atmos: atmospheric timestep in seconds
    """

    stencil_config: pace.dsl.StencilConfig
    initialization_type: str
    initialization_config: InitializationConfig
    nx_tile: int
    nz: int
    layout: Tuple[int, int]
    dt_atmos: float
    diagnostics_config: DiagnosticsConfig
    dycore_config: fv3core.DynamicalCoreConfig = dataclasses.field(
        default_factory=fv3core.DynamicalCoreConfig
    )
    physics_config: fv3gfs.physics.PhysicsConfig = dataclasses.field(
        default_factory=fv3gfs.physics.PhysicsConfig
    )
    days: int = 0
    hours: int = 0
    minutes: int = 0
    seconds: int = 0

    @functools.cached_property
    def timestep(self) -> timedelta:
        return timedelta(seconds=self.dt_atmos)

    @property
    def start_time(self) -> Union[datetime, timedelta]:
        return self.initialization_config.start_time

    @functools.cached_property
    def total_time(self) -> timedelta:
        return timedelta(
            days=self.days, hours=self.hours, minutes=self.minutes, seconds=self.seconds
        )

    @classmethod
    def from_dict(cls, kwargs: Dict[str, Any]) -> "DriverConfig":
        kwargs["layout"] = tuple(kwargs["layout"])
        initialization_type = kwargs["initialization_type"]
        if initialization_type == "baroclinic":
            initialization_class = BaroclinicConfig
        elif initialization_type == "restart":
            initialization_class = RestartConfig
        else:
            raise ValueError(
                "initialization_type must be one of 'baroclinic' or 'restart', "
                f"got {initialization_type}"
            )
        kwargs["initialization_config"] = dacite.from_dict(
            data_class=initialization_class,
            data=kwargs.get("initialization_config", {}),
            config=dacite.Config(strict=True),
        )
        for derived_name in ("dt_atmos", "layout", "npx", "npy", "npz", "ntiles"):
            if derived_name in kwargs["dycore_config"]:
                raise ValueError(
                    f"you cannot set {derived_name} directly in dycore_config, "
                    "as it is determined based on top-level configuration"
                )
        kwargs["dycore_config"]["layout"] = kwargs["layout"]
        kwargs["dycore_config"]["dt_atmos"] = kwargs["dt_atmos"]
        kwargs["dycore_config"]["npx"] = kwargs["nx_tile"] + 1
        kwargs["dycore_config"]["npy"] = kwargs["nx_tile"] + 1
        kwargs["dycore_config"]["npz"] = kwargs["nz"]
        kwargs["dycore_config"]["ntiles"] = 6
        for derived_name in ("dt_atmos", "layout", "npx", "npy", "npz"):
            if derived_name in kwargs["physics_config"]:
                raise ValueError(
                    f"you cannot set {derived_name} directly in physics_config, "
                    "as it is determined based on top-level configuration"
                )
        kwargs["physics_config"]["layout"] = kwargs["layout"]
        kwargs["physics_config"]["dt_atmos"] = kwargs["dt_atmos"]
        kwargs["physics_config"]["npx"] = kwargs["nx_tile"] + 1
        kwargs["physics_config"]["npy"] = kwargs["nx_tile"] + 1
        kwargs["physics_config"]["npz"] = kwargs["nz"]
        return dacite.from_dict(
            data_class=cls, data=kwargs, config=dacite.Config(strict=True)
        )


class Driver:
    def __init__(
        self,
        config: DriverConfig,
        comm,
    ):
        """
        Initializes a pace Driver.

        Args:
            config: driver configuration
            comm: communication object behaving like mpi4py.Comm
        """
        self.config = config
        communicator = pace.util.CubedSphereCommunicator.from_layout(
            comm=comm, layout=self.config.layout
        )
        quantity_factory, stencil_factory = _setup_factories(
            config=config, communicator=communicator
        )
        metric_terms = pace.util.grid.MetricTerms(
            quantity_factory=quantity_factory, communicator=communicator
        )
        grid_data = pace.util.grid.GridData.new_from_metric_terms(metric_terms)
        self.state = self.config.initialization_config.get_driver_state(
            quantity_factory=quantity_factory, communicator=communicator
        )
        self._start_time = self.config.initialization_config.start_time
        self.dycore = fv3core.DynamicalCore(
            comm=communicator,
            grid_data=grid_data,
            stencil_factory=stencil_factory,
            damping_coefficients=DampingCoefficients.new_from_metric_terms(
                metric_terms
            ),
            config=self.config.dycore_config,
            phis=self.state.dycore_state.phis,
        )
        self.physics = fv3gfs.physics.Physics(
            stencil_factory=stencil_factory,
            grid_data=grid_data,
            namelist=self.config.physics_config,
            active_packages=["microphysics"],
        )
        self.dycore_to_physics = update_atmos_state.DycoreToPhysics(
            stencil_factory=stencil_factory
        )
        self.physics_to_dycore = update_atmos_state.UpdateAtmosphereState(
            stencil_factory=stencil_factory,
            grid_data=grid_data,
            namelist=self.config.physics_config,
            comm=communicator,
            grid_info=pace.util.grid.DriverGridData.new_from_metric_terms(metric_terms),
            quantity_factory=quantity_factory,
        )
        self.diagnostics = Diagnostics(
            config=config.diagnostics_config,
            partitioner=communicator.partitioner,
            comm=comm,
        )

    def step_all(self):
        time = self.config.start_time
        end_time = self.config.start_time + self.config.total_time
        self.diagnostics.store(time=time, state=self.state)
        while time < end_time:
            self._step(timestep=self.config.timestep.total_seconds())
            time += self.config.timestep
            self.diagnostics.store(time=time, state=self.state)

    def _step(self, timestep: float):
        self._step_dynamics(timestep=timestep)
        self._step_physics(timestep=timestep)

    def _step_dynamics(self, timestep: float):
        self.dycore.step_dynamics(
            state=self.state.dycore_state,
            conserve_total_energy=self.config.dycore_config.consv_te,
            n_split=self.config.dycore_config.n_split,
            do_adiabatic_init=False,
            timestep=float(timestep),
        )

    def _step_physics(self, timestep: float):
        self.dycore_to_physics(
            dycore_state=self.state.dycore_state, physics_state=self.state.physics_state
        )
        self.physics(self.state.physics_state, timestep=float(timestep))
        self.physics_to_dycore(
            dycore_state=self.state.dycore_state,
            phy_state=self.state.physics_state,
            dt=float(timestep),
        )


def _setup_factories(
    config: DriverConfig, communicator: pace.util.CubedSphereCommunicator
) -> Tuple["pace.util.QuantityFactory", "pace.dsl.StencilFactory"]:
    sizer = pace.util.SubtileGridSizer.from_tile_params(
        nx_tile=config.nx_tile,
        ny_tile=config.nx_tile,
        nz=config.nz,
        n_halo=pace.util.N_HALO_DEFAULT,
        extra_dim_lengths={},
        layout=config.layout,
        tile_partitioner=communicator.partitioner.tile,
        tile_rank=communicator.tile.rank,
    )

    grid_indexing = pace.dsl.stencil.GridIndexing.from_sizer_and_communicator(
        sizer=sizer, cube=communicator
    )
    quantity_factory = pace.util.QuantityFactory.from_backend(
        sizer, backend=config.stencil_config.backend
    )
    stencil_factory = pace.dsl.StencilFactory(
        config=config.stencil_config,
        grid_indexing=grid_indexing,
    )
    return quantity_factory, stencil_factory


@click.command()
@click.argument(
    "config_path",
    required=True,
)
def command_line(config_path: str):
    with open(config_path, "r") as f:
        driver_config = DriverConfig.from_dict(yaml.safe_load(f))
    main(driver_config=driver_config, comm=MPI.COMM_WORLD)


def main(driver_config: DriverConfig, comm):
    driver = Driver(
        config=driver_config,
        comm=comm,
    )
    driver.step_all()


if __name__ == "__main__":
    command_line()
