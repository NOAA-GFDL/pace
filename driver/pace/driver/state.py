import dataclasses
from dataclasses import fields

import xarray as xr

import pace.dsl.gt4py_utils as gt_utils
import pace.physics
from pace import fv3core
from pace.dsl.typing import Float
from pace.util.grid.helper import DampingCoefficients, DriverGridData, GridData


@dataclasses.dataclass()
class TendencyState:
    """
    Accumulated tendencies from physical parameterizations to be applied
    to the dynamical core model state.
    """

    u_dt: pace.util.Quantity = dataclasses.field(
        metadata={
            "name": "eastward_wind_tendency_due_to_physics",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s**2",
            "intent": "inout",
        }
    )
    v_dt: pace.util.Quantity = dataclasses.field(
        metadata={
            "name": "northward_wind_tendency_due_to_physics",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s**2",
            "intent": "inout",
        }
    )
    pt_dt: pace.util.Quantity = dataclasses.field(
        metadata={
            "name": "temperature_tendency_due_to_physics",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "K/s",
            "intent": "inout",
        }
    )

    @classmethod
    def init_zeros(cls, quantity_factory: pace.util.QuantityFactory) -> "TendencyState":
        initial_quantities = {}
        for _field in dataclasses.fields(cls):
            initial_quantities[_field.name] = quantity_factory.zeros(
                _field.metadata["dims"],
                _field.metadata["units"],
                dtype=Float,
            )
        return cls(**initial_quantities)


@dataclasses.dataclass
class DriverState:
    dycore_state: fv3core.DycoreState
    physics_state: pace.physics.PhysicsState
    tendency_state: TendencyState
    grid_data: GridData
    damping_coefficients: DampingCoefficients
    driver_grid_data: DriverGridData

    # TODO: the driver_config argument here isn't type hinted from
    # import due to a circular dependency. This can be fixed by refactoring
    # for example by moving this method into some restart.py module
    @classmethod
    def load_state_from_restart(
        cls,
        restart_path: str,
        driver_config,
        damping_coefficients: DampingCoefficients,
        driver_grid_data: DriverGridData,
        grid_data: GridData,
    ) -> "DriverState":
        comm = driver_config.comm_config.get_comm()
        communicator = pace.util.Communicator.from_layout(
            comm=comm, layout=driver_config.layout
        )
        sizer = pace.util.SubtileGridSizer.from_tile_params(
            nx_tile=driver_config.nx_tile,
            ny_tile=driver_config.nx_tile,
            nz=driver_config.nz,
            n_halo=pace.util.N_HALO_DEFAULT,
            extra_dim_lengths={},
            layout=driver_config.layout,
            tile_partitioner=communicator.partitioner.tile,
            tile_rank=communicator.tile.rank,
        )
        quantity_factory = pace.util.QuantityFactory.from_backend(
            sizer, backend=driver_config.stencil_config.compilation_config.backend
        )

        state = _restart_driver_state(
            restart_path,
            communicator.rank,
            quantity_factory,
            communicator,
            damping_coefficients=damping_coefficients,
            driver_grid_data=driver_grid_data,
            grid_data=grid_data,
        )
        return state

    def save_state(self, comm, restart_path: str = "RESTART"):
        from pathlib import Path

        Path(restart_path).mkdir(parents=True, exist_ok=True)
        current_rank = str(comm.Get_rank())
        self.dycore_state.xr_dataset.to_netcdf(
            f"{restart_path}/restart_dycore_state_{current_rank}.nc"
        )
        self.physics_state.xr_dataset.to_netcdf(
            f"{restart_path}/restart_physics_state_{current_rank}.nc"
        )
        # we can also convert the state to Fortran's restart format using
        # code similar to this commented code. We don't need this feature right
        # now so we haven't implemented it, but this is a good starter.
        """
        xr.Dataset(
            data_vars={
                "cld_amt": state.dycore_state.qcld.data_array,
                "graupel": state.dycore_state.qgraupel.data_array,
                "ice_wat": state.dycore_state.qice.data_array,
                "liq_wat": state.dycore_state.qliquid.data_array,
                "o3mr": state.dycore_state.qo3mr.data_array,
                "rainwat": state.dycore_state.qrain.data_array,
                "sgs_tke": state.dycore_state.qsgs_tke.data_array,
                "snowwat": state.dycore_state.qsnow.data_array,
                "sphum": state.dycore_state.qvapor.data_array,
            }
        ).rename(
            {
                "z": "zaxis_1",
                "x": "xaxis_1",
                "y": "yaxis_1",
            }
        ).transpose(
            "zaxis_1", "yaxis_1", "xaxis_1"
        ).expand_dims(
            dim="Time", axis=0
        ).to_netcdf(os.path.join(path, f"fv_tracer.res.tile{rank + 1}.nc"))
        """


def _overwrite_state_from_restart(
    path: str,
    rank: int,
    state: fv3core.DycoreState,
    restart_file_prefix: str,
):
    """
    Args:
        path: path to restart files
        rank: current rank number
        state: an empty state
        restart_file_prefix: file prefix name to read
    """
    ds = xr.open_dataset(path + f"/{restart_file_prefix}_{rank}.nc")

    for _field in fields(type(state)):
        if "units" in _field.metadata.keys():
            state.__dict__[_field.name].data[:] = gt_utils.asarray(
                ds[_field.name].data[:], to_type=state.__dict__[_field.name].np.ndarray
            )


def _restart_driver_state(
    path: str,
    rank: int,
    quantity_factory: pace.util.QuantityFactory,
    communicator: pace.util.Communicator,
    damping_coefficients: DampingCoefficients,
    driver_grid_data: DriverGridData,
    grid_data: GridData,
):
    fs = pace.util.get_fs(path)

    restart_files = fs.ls(path)
    is_fortran_restart = any(
        fname.endswith("fv_core.res.nc") for fname in restart_files
    )

    if is_fortran_restart:
        dycore_state = fv3core.DycoreState.from_fortran_restart(
            quantity_factory=quantity_factory, communicator=communicator, path=path
        )
    else:
        dycore_state = fv3core.DycoreState.init_zeros(quantity_factory=quantity_factory)
        _overwrite_state_from_restart(
            path,
            rank,
            dycore_state,
            "restart_dycore_state",
        )

    active_packages = ["microphysics"]
    physics_state = pace.physics.PhysicsState.init_zeros(
        quantity_factory=quantity_factory, active_packages=active_packages
    )

    physics_state.__post_init__(quantity_factory, active_packages)
    tendency_state = TendencyState.init_zeros(
        quantity_factory=quantity_factory,
    )

    return DriverState(
        dycore_state=dycore_state,
        physics_state=physics_state,
        tendency_state=tendency_state,
        grid_data=grid_data,
        damping_coefficients=damping_coefficients,
        driver_grid_data=driver_grid_data,
    )
