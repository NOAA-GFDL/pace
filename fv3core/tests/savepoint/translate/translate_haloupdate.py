from ndsl.constants import (
    N_HALO_DEFAULT,
    X_DIM,
    X_INTERFACE_DIM,
    Y_DIM,
    Y_INTERFACE_DIM,
    Z_DIM,
    Z_INTERFACE_DIM,
)
from ndsl.dsl.stencil import StencilFactory
from ndsl.logging import ndsl_log
from ndsl.namelist import Namelist
from ndsl.stencils.testing import ParallelTranslate


class TranslateHaloUpdate(ParallelTranslate):
    inputs = {
        "array": {
            "name": "air_temperature",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "degK",
            "n_halo": N_HALO_DEFAULT,
        }
    }

    outputs = {
        "array": {
            "name": "air_temperature",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "degK",
            "n_halo": N_HALO_DEFAULT,
        }
    }
    halo_update_varname = "air_temperature"

    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)

    def compute_parallel(self, inputs, communicator):
        state = self.state_from_inputs(inputs)
        req = communicator.start_halo_update(
            state[self.halo_update_varname], n_points=N_HALO_DEFAULT
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
                    state[self.halo_update_varname], n_points=N_HALO_DEFAULT
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
            "dims": [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            "units": "m",
            "n_halo": N_HALO_DEFAULT,
        }
    }

    outputs = {
        "array2": {
            "name": "height_on_interface_levels",
            "dims": [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            "units": "m",
            "n_halo": N_HALO_DEFAULT,
        }
    }

    halo_update_varname = "height_on_interface_levels"


class TranslateMPPUpdateDomains(TranslateHaloUpdate):
    inputs = {
        "update_arr": {
            "name": "z_wind_as_tendency_of_pressure",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "Pa/s",
            "n_halo": N_HALO_DEFAULT,
        }
    }

    outputs = {
        "update_arr": {
            "name": "z_wind_as_tendency_of_pressure",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "Pa/s",
            "n_halo": N_HALO_DEFAULT,
        }
    }

    halo_update_varname = "z_wind_as_tendency_of_pressure"


class TranslateHaloVectorUpdate(ParallelTranslate):
    inputs = {
        "array_u": {
            "name": "x_wind_on_c_grid",
            "dims": [X_INTERFACE_DIM, Y_DIM, Z_DIM],
            "units": "m/s",
            "n_halo": N_HALO_DEFAULT,
        },
        "array_v": {
            "name": "y_wind_on_c_grid",
            "dims": [X_DIM, Y_INTERFACE_DIM, Z_DIM],
            "units": "m/s",
            "n_halo": N_HALO_DEFAULT,
        },
    }

    outputs = {
        "array_u": {
            "name": "x_wind_on_c_grid",
            "dims": [X_INTERFACE_DIM, Y_DIM, Z_DIM],
            "units": "m/s",
            "n_halo": N_HALO_DEFAULT,
        },
        "array_v": {
            "name": "y_wind_on_c_grid",
            "dims": [X_DIM, Y_INTERFACE_DIM, Z_DIM],
            "units": "m/s",
            "n_halo": N_HALO_DEFAULT,
        },
    }

    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super(TranslateHaloVectorUpdate, self).__init__(grid, namelist, stencil_factory)

    def compute_parallel(self, inputs, communicator):
        ndsl_log.debug(f"starting on {communicator.rank}")
        state = self.state_from_inputs(inputs)
        req = communicator.start_vector_halo_update(
            state["x_wind_on_c_grid"],
            state["y_wind_on_c_grid"],
            n_points=N_HALO_DEFAULT,
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
                    n_points=N_HALO_DEFAULT,
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
            "dims": [X_DIM, Y_INTERFACE_DIM, Z_DIM],
            "units": "m/s",
            "n_halo": N_HALO_DEFAULT,
        },
        "v": {
            "name": "y_wind_on_d_grid",
            "dims": [X_INTERFACE_DIM, Y_DIM, Z_DIM],
            "units": "m/s",
            "n_halo": N_HALO_DEFAULT,
        },
    }

    outputs = {
        "u": {
            "name": "x_wind_on_d_grid",
            "dims": [X_DIM, Y_INTERFACE_DIM, Z_DIM],
            "units": "m/s",
            "n_halo": N_HALO_DEFAULT,
        },
        "v": {
            "name": "y_wind_on_d_grid",
            "dims": [X_INTERFACE_DIM, Y_DIM, Z_DIM],
            "units": "m/s",
            "n_halo": N_HALO_DEFAULT,
        },
    }

    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
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
