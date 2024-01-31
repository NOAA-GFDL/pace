import ndsl.dsl
import ndsl.util
import ndsl.util as fv3util
from ndsl.stencils.testing import ParallelTranslate
from ndsl.util.logging import ndsl_log


class TranslateHaloUpdate(ParallelTranslate):
    inputs = {
        "array": {
            "name": "air_temperature",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "degK",
            "n_halo": ndsl.util.N_HALO_DEFAULT,
        }
    }

    outputs = {
        "array": {
            "name": "air_temperature",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "degK",
            "n_halo": ndsl.util.N_HALO_DEFAULT,
        }
    }
    halo_update_varname = "air_temperature"

    def __init__(
        self,
        grid,
        namelist: ndsl.util.Namelist,
        stencil_factory: ndsl.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)

    def compute_parallel(self, inputs, communicator):
        state = self.state_from_inputs(inputs)
        req = communicator.start_halo_update(
            state[self.halo_update_varname], n_points=fv3util.N_HALO_DEFAULT
        )
        req.wait()
        return self.outputs_from_state(state)

    def compute_sequential(self, inputs_list, communicator_list):
        state_list = self.state_list_from_inputs_list(inputs_list)
        req_list = []
        for state, communicator in zip(state_list, communicator_list):
            ndsl_log.debug(f"starting on {communicator.rank}")
            req_list.append(
                communicator.start_halo_update(
                    state[self.halo_update_varname], n_points=fv3util.N_HALO_DEFAULT
                )
            )
        for communicator, req in zip(communicator_list, req_list):
            ndsl_log.debug(f"finishing on {communicator.rank}")
            req.wait()
        return self.outputs_list_from_state_list(state_list)


class TranslateHaloUpdate_2(TranslateHaloUpdate):
    inputs = {
        "array2": {
            "name": "height_on_interface_levels",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM],
            "units": "m",
            "n_halo": fv3util.N_HALO_DEFAULT,
        }
    }

    outputs = {
        "array2": {
            "name": "height_on_interface_levels",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM],
            "units": "m",
            "n_halo": fv3util.N_HALO_DEFAULT,
        }
    }

    halo_update_varname = "height_on_interface_levels"


class TranslateMPPUpdateDomains(TranslateHaloUpdate):
    inputs = {
        "update_arr": {
            "name": "z_wind_as_tendency_of_pressure",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "Pa/s",
            "n_halo": fv3util.N_HALO_DEFAULT,
        }
    }

    outputs = {
        "update_arr": {
            "name": "z_wind_as_tendency_of_pressure",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "Pa/s",
            "n_halo": fv3util.N_HALO_DEFAULT,
        }
    }

    halo_update_varname = "z_wind_as_tendency_of_pressure"


class TranslateHaloVectorUpdate(ParallelTranslate):
    inputs = {
        "array_u": {
            "name": "x_wind_on_c_grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": fv3util.N_HALO_DEFAULT,
        },
        "array_v": {
            "name": "y_wind_on_c_grid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": fv3util.N_HALO_DEFAULT,
        },
    }

    outputs = {
        "array_u": {
            "name": "x_wind_on_c_grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": fv3util.N_HALO_DEFAULT,
        },
        "array_v": {
            "name": "y_wind_on_c_grid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": fv3util.N_HALO_DEFAULT,
        },
    }

    def __init__(
        self,
        grid,
        namelist: ndsl.util.Namelist,
        stencil_factory: ndsl.dsl.StencilFactory,
    ):
        super(TranslateHaloVectorUpdate, self).__init__(grid, namelist, stencil_factory)

    def compute_parallel(self, inputs, communicator):
        ndsl_log.debug(f"starting on {communicator.rank}")
        state = self.state_from_inputs(inputs)
        req = communicator.start_vector_halo_update(
            state["x_wind_on_c_grid"],
            state["y_wind_on_c_grid"],
            n_points=fv3util.N_HALO_DEFAULT,
        )

        ndsl_log.debug(f"finishing on {communicator.rank}")
        req.wait()
        return self.outputs_from_state(state)

    def compute_sequential(self, inputs_list, communicator_list):
        state_list = self.state_list_from_inputs_list(inputs_list)
        req_list = []
        for state, communicator in zip(state_list, communicator_list):
            ndsl_log.debug(f"starting on {communicator.rank}")
            req_list.append(
                communicator.start_vector_halo_update(
                    state["x_wind_on_c_grid"],
                    state["y_wind_on_c_grid"],
                    n_points=fv3util.N_HALO_DEFAULT,
                )
            )
        for communicator, req in zip(communicator_list, req_list):
            ndsl_log.debug(f"finishing on {communicator.rank}")
            req.wait()
        return self.outputs_list_from_state_list(state_list)


class TranslateMPPBoundaryAdjust(ParallelTranslate):
    inputs = {
        "u": {
            "name": "x_wind_on_d_grid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": fv3util.N_HALO_DEFAULT,
        },
        "v": {
            "name": "y_wind_on_d_grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": fv3util.N_HALO_DEFAULT,
        },
    }

    outputs = {
        "u": {
            "name": "x_wind_on_d_grid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": fv3util.N_HALO_DEFAULT,
        },
        "v": {
            "name": "y_wind_on_d_grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": fv3util.N_HALO_DEFAULT,
        },
    }

    def __init__(
        self,
        grid,
        namelist: ndsl.util.Namelist,
        stencil_factory: ndsl.dsl.StencilFactory,
    ):
        super(TranslateMPPBoundaryAdjust, self).__init__(
            grid, namelist, stencil_factory
        )

    def compute_parallel(self, inputs, communicator):
        ndsl_log.debug(f"starting on {communicator.rank}")
        state = self.state_from_inputs(inputs)
        req = communicator.start_synchronize_vector_interfaces(
            state["x_wind_on_d_grid"], state["y_wind_on_d_grid"]
        )
        ndsl_log.debug(f"finishing on {communicator.rank}")
        req.wait()
        return self.outputs_from_state(state)

    def compute_sequential(self, inputs_list, communicator_list):
        state_list = self.state_list_from_inputs_list(inputs_list)
        req_list = []
        for state, communicator in zip(state_list, communicator_list):
            req_list.append(
                communicator.start_synchronize_vector_interfaces(
                    state["x_wind_on_d_grid"], state["y_wind_on_d_grid"]
                )
            )
        for communicator, req in zip(communicator_list, req_list):
            ndsl_log.debug(f"finishing on {communicator.rank}")
            req.wait()
        return self.outputs_list_from_state_list(state_list)
