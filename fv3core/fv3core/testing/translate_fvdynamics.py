from dataclasses import fields
from typing import Any, Dict, Optional

import pytest

import fv3core._config as spec
import fv3core.stencils.fv_dynamics as fv_dynamics
import pace.dsl.gt4py_utils as utils
import pace.util as fv3util
from fv3core.initialization.dycore_state import DycoreState
from fv3core.testing import ParallelTranslateBaseSlicing


ADVECTED_TRACER_NAMES = utils.tracer_variables[: fv_dynamics.NQ]


class TranslateFVDynamics(ParallelTranslateBaseSlicing):
    python_regression = True
    compute_grid_option = True
    inputs: Dict[str, Any] = {
        "q_con": {
            "name": "total_condensate_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "delp": {
            "name": "pressure_thickness_of_atmospheric_layer",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "Pa",
        },
        "delz": {
            "name": "vertical_thickness_of_atmospheric_layer",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m",
        },
        "ps": {
            "name": "surface_pressure",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "Pa",
        },
        "pe": {
            "name": "interface_pressure",
            "dims": [fv3util.X_DIM, fv3util.Z_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "Pa",
            "n_halo": 1,
        },
        "ak": {
            "name": "atmosphere_hybrid_a_coordinate",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "Pa",
        },
        "bk": {
            "name": "atmosphere_hybrid_b_coordinate",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "",
        },
        "pk": {
            "name": "interface_pressure_raised_to_power_of_kappa",
            "units": "unknown",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM],
            "n_halo": 0,
        },
        "pkz": {
            "name": "layer_mean_pressure_raised_to_power_of_kappa",
            "units": "unknown",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "n_halo": 0,
        },
        "peln": {
            "name": "logarithm_of_interface_pressure",
            "units": "ln(Pa)",
            "dims": [fv3util.X_DIM, fv3util.Z_INTERFACE_DIM, fv3util.Y_DIM],
            "n_halo": 0,
        },
        "mfxd": {
            "name": "accumulated_x_mass_flux",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "unknown",
            "n_halo": 0,
        },
        "mfyd": {
            "name": "accumulated_y_mass_flux",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "unknown",
            "n_halo": 0,
        },
        "cxd": {
            "name": "accumulated_x_courant_number",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "",
            "n_halo": (0, 3),
        },
        "cyd": {
            "name": "accumulated_y_courant_number",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "",
            "n_halo": (3, 0),
        },
        "diss_estd": {
            "name": "dissipation_estimate_from_heat_source",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "unknown",
        },
        "pt": {
            "name": "air_temperature",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "degK",
        },
        "u": {
            "name": "x_wind",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "v": {
            "name": "y_wind",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "ua": {
            "name": "eastward_wind",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "va": {
            "name": "northward_wind",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "uc": {
            "name": "x_wind_on_c_grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "vc": {
            "name": "y_wind_on_c_grid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "w": {
            "name": "vertical_wind",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "phis": {
            "name": "surface_geopotential",
            "units": "m^2 s^-2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
        },
        "qvapor": {
            "name": "specific_humidity",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qliquid": {
            "name": "cloud_water_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qice": {
            "name": "cloud_ice_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qrain": {
            "name": "rain_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qsnow": {
            "name": "snow_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qgraupel": {
            "name": "graupel_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qo3mr": {
            "name": "ozone_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qsgs_tke": {
            "name": "turbulent_kinetic_energy",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m**2/s**2",
        },
        "qcld": {
            "name": "cloud_fraction",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "",
        },
        "omga": {
            "name": "vertical_pressure_velocity",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "Pa/s",
        },
        "do_adiabatic_init": {"dims": []},
        "bdt": {"dims": []},
        "ptop": {"dims": []},
        "ks": {"dims": []},
    }

    outputs = inputs.copy()

    for name in ("do_adiabatic_init", "bdt", "ak", "bk", "ks", "ptop"):
        outputs.pop(name)

    def __init__(self, grids, *args, **kwargs):
        super().__init__(grids, *args, **kwargs)
        grid = grids[0]
        self._base.in_vars["data_vars"] = {
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "w": {},
            "delz": {},
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qo3mr": {},
            "qsgs_tke": {},
            "qcld": {},
            "ps": {},
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "pk": grid.compute_buffer_k_dict(),
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
                "kaxis": 1,
            },
            "pkz": grid.compute_dict(),
            "phis": {},
            "q_con": {},
            "delp": {},
            "pt": {},
            "omga": {},
            "ua": {},
            "va": {},
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "mfxd": grid.x3d_compute_dict(),
            "mfyd": grid.y3d_compute_dict(),
            "cxd": grid.x3d_compute_domain_y_dict(),
            "cyd": grid.y3d_compute_domain_x_dict(),
            "diss_estd": {},
        }

        self._base.out_vars = self._base.in_vars["data_vars"].copy()
        self._base.out_vars["ps"] = {"kstart": grid.npz - 1, "kend": grid.npz - 1}
        self._base.out_vars["phis"] = {"kstart": grid.npz - 1, "kend": grid.npz - 1}

        self.max_error = 1e-5

        self.ignore_near_zero_errors = {}
        for qvar in utils.tracer_variables:
            self.ignore_near_zero_errors[qvar] = True
        self.ignore_near_zero_errors["q_con"] = True
        self.dycore: Optional[fv_dynamics.DynamicalCore] = None

    def state_from_inputs(self, inputs):
        input_storages = super().state_from_inputs(inputs)
        # making sure we init DycoreState with the exact set of variables
        accepted_keys = [_field.name for _field in fields(DycoreState)]
        todelete = []
        for name, quantity in input_storages.items():
            if name not in accepted_keys:
                todelete.append(name)
        for name in todelete:
            del input_storages[name]

        state = DycoreState(
            **input_storages, quantity_factory=self.grid.quantity_factory
        )
        return state

    def compute_parallel(self, inputs, communicator):
        for name in ("ak", "bk"):
            inputs[name] = utils.make_storage_data(
                inputs[name],
                inputs[name].shape,
                len(inputs[name].shape) * (0,),
                backend=self.grid.stencil_factory.backend,
            )
        grid_data = spec.grid.grid_data
        # These aren't in the Grid-Info savepoint, but are in the generated grid
        if grid_data.ak is None or grid_data.bk is None:
            grid_data.ak = inputs["ak"]
            grid_data.bk = inputs["bk"]
            grid_data.ptop = inputs["ptop"]
            grid_data.ks = inputs["ks"]

        state = self.state_from_inputs(inputs)
        self.dycore = fv_dynamics.DynamicalCore(
            comm=communicator,
            grid_data=grid_data,
            stencil_factory=spec.grid.stencil_factory,
            damping_coefficients=spec.grid.damping_coefficients,
            config=spec.namelist.dynamical_core,
            phis=state.phis,
        )
        self.dycore.step_dynamics(
            state,
            spec.namelist.consv_te,
            inputs["do_adiabatic_init"],
            inputs["bdt"],
            spec.namelist.n_split,
        )
        outputs = self.outputs_from_state(state)
        for name, value in outputs.items():
            outputs[name] = self.subset_output(name, value)
        return outputs

    def outputs_from_state(self, state: dict):
        if len(self.outputs) == 0:
            return {}
        outputs = {}
        storages = {}
        for name, properties in self.outputs.items():
            if isinstance(state[name], fv3util.Quantity):
                storages[name] = state[name].storage
            elif len(self.outputs[name]["dims"]) > 0:
                storages[name] = state[name]  # assume it's a storage
            else:
                outputs[name] = state[name]  # scalar
        outputs.update(self._base.slice_output(storages))
        return outputs

    def compute_sequential(self, *args, **kwargs):
        pytest.skip(
            f"{self.__class__} only has a mpirun implementation, "
            "not running in mock-parallel"
        )

    def subset_output(self, varname: str, output):
        """
        Given an output array, return the slice of the array which we'd
        like to validate against reference data
        """
        if self.dycore is None:
            raise RuntimeError(
                "cannot call subset_output before calling compute_parallel "
                "to initialize dycore"
            )
        elif varname in self.dycore.selective_names:  # type: ignore
            return_value = self.dycore.subset_output(varname, output)  # type: ignore
        elif varname in ADVECTED_TRACER_NAMES:
            return_value = self.dycore.tracer_advection.subset_output(  # type: ignore
                "tracers", output
            )
        else:
            return_value = output
        return return_value


# Method for creating a DycoreState object from serialized data
def init_dycore_state_from_serialized_data(serializer, grid, quantity_factory):
    savepoint_in = serializer.get_savepoint("FVDynamics-In")[0]
    translate_object = TranslateFVDynamics([grid])
    input_data = translate_object.collect_input_data(serializer, savepoint_in)
    # making just storages for the moment, revisit when making them all
    # quantities (maybe use state_from_inputs)
    translate_object._base.make_storage_data_input_vars(input_data)
    # used for the translate test as inputs, but are generated by the
    # MetricsTerms class and are not part of this data class
    for delvar in ["ak", "bk", "ptop", "ks"]:
        del input_data[delvar]
    return DycoreState(**input_data, quantity_factory=quantity_factory)
