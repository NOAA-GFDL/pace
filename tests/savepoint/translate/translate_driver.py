from ndsl.constants import N_HALO_DEFAULT
from ndsl.initialization.allocator import QuantityFactory
from ndsl.initialization.sizer import SubtileGridSizer
from ndsl.namelist import Namelist
from pace.driver.run import Driver, DriverConfig
from pace.driver.state import TendencyState
from pace.fv3core._config import DynamicalCoreConfig

# TODO physics should not depend on fv3core
# but also, driver tests should not be in physics
from pace.fv3core.testing.translate_fvdynamics import TranslateFVDynamics
from pace.fv3core.testing.validation import enable_selective_validation
from pace.physics import PHYSICS_PACKAGES, PhysicsConfig, PhysicsState


enable_selective_validation()


class TranslateDriver(TranslateFVDynamics):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.namelist: Namelist = namelist
        self.stencil_factory = stencil_factory
        self.stencil_config = self.stencil_factory.config

        # TODO: use threshold calibration to set this properly
        # increase this for circleci tests
        self.max_error = 3e-5

    def compute_parallel(self, inputs, communicator):
        dycore_state = self.state_from_inputs(inputs)
        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npy - 1,
            nz=self.namelist.npz,
            n_halo=N_HALO_DEFAULT,
            extra_dim_lengths={},
            layout=self.namelist.layout,
            tile_partitioner=communicator.partitioner.tile,
            tile_rank=communicator.tile.rank,
        )

        quantity_factory = QuantityFactory.from_backend(
            sizer, backend=self.stencil_config.compilation_config.backend
        )
        physics_state = PhysicsState.init_zeros(
            quantity_factory=quantity_factory,
            schemes=[PHYSICS_PACKAGES["GFS_microphysics"]],
        )
        tendency_state = TendencyState.init_zeros(
            quantity_factory=quantity_factory,
        )
        config_info = {
            "stencil_config": self.stencil_config,
            "initialization": {
                "type": "predefined",
                "config": {
                    "dycore_state": dycore_state,
                    "grid_data": self.grid.grid_data,
                    "damping_coefficients": self.grid.damping_coefficients,
                    "driver_grid_data": self.grid.driver_grid_data,
                    "physics_state": physics_state,
                    "tendency_state": tendency_state,
                },
            },
            "dt_atmos": self.namelist.dt_atmos,
            "diagnostics_config": {"path": "null.zarr", "names": []},
            "performance_config": {"collect_performance": False},
            "dycore_config": DynamicalCoreConfig.from_namelist(self.namelist),
            "physics_config": PhysicsConfig.from_namelist(self.namelist),
            "seconds": self.namelist.dt_atmos,
            "dycore_only": self.namelist.dycore_only,
            "nx_tile": self.namelist.npx - 1,
            "nz": self.namelist.npz,
            "layout": tuple(self.namelist.layout),
        }
        config = DriverConfig.from_dict(config_info)
        driver = Driver(config=config)

        driver.step_all()
        self.dycore = driver.dycore

        outputs = self.outputs_from_state(driver.state.dycore_state)
        return outputs
