import pytest

import fv3core._config as spec
import fv3core.stencils.fv_dynamics as fv_dynamics
import fv3core.stencils.fvtp2d
import fv3core.stencils.tracer_2d_1l
import fv3gfs.util as fv3util
import fv3gfs.util.pace.gt4py_utils as utils
from fv3core.testing import ParallelTranslate


class TranslateTracer2D1L(ParallelTranslate):
    inputs = {
        "tracers": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/m^2",
        }
    }

    def __init__(self, grids):
        super().__init__(grids)
        grid = grids[0]
        self._base.in_vars["data_vars"] = {
            "tracers": {},
            "dp1": {},
            "mfxd": grid.x3d_compute_dict(),
            "mfyd": grid.y3d_compute_dict(),
            "cxd": grid.x3d_compute_domain_y_dict(),
            "cyd": grid.y3d_compute_domain_x_dict(),
        }
        self._base.in_vars["parameters"] = ["nq", "mdt"]
        self._base.out_vars = self._base.in_vars["data_vars"]

    def collect_input_data(self, serializer, savepoint):
        input_data = self._base.collect_input_data(serializer, savepoint)
        return input_data

    def compute_parallel(self, inputs, communicator):

        self._base.make_storage_data_input_vars(inputs)
        all_tracers = inputs["tracers"]
        inputs["tracers"] = self.get_advected_tracer_dict(
            inputs["tracers"], inputs.pop("nq")
        )
        transport = fv3core.stencils.fvtp2d.FiniteVolumeTransport(
            stencil_factory=spec.grid.stencil_factory,
            grid_data=spec.grid.grid_data,
            damping_coefficients=spec.grid.damping_coefficients,
            grid_type=spec.grid.grid_type,
            hord=spec.namelist.hord_tr,
        )
        namelist = spec.namelist

        self.tracer_advection = fv3core.stencils.tracer_2d_1l.TracerAdvection(
            self.grid.stencil_factory,
            transport,
            spec.grid.grid_data,
            communicator,
            fv_dynamics.NQ,
        )
        self.tracer_advection(**inputs)
        inputs[
            "tracers"
        ] = all_tracers  # some aren't advected, still need to be validated
        # need to convert tracers dict to [x, y, z, n_tracer] array before subsetting
        outputs = self._base.slice_output(inputs)
        outputs["tracers"] = self.subset_output("tracers", outputs["tracers"])
        return outputs

    def get_advected_tracer_dict(self, all_tracers, nq):
        all_tracers = {**all_tracers}  # make a new dict so we don't modify the input
        properties = self.inputs["tracers"]
        for name in utils.tracer_variables:
            self.grid.quantity_dict_update(
                all_tracers,
                name,
                dims=properties["dims"],
                units=properties["units"],
            )
        tracer_names = utils.tracer_variables[:nq]
        return {name: all_tracers[name + "_quantity"] for name in tracer_names}

    def compute_sequential(self, a, b):
        pytest.skip(
            f"{self.__class__} only has a mpirun implementation, "
            "not running in mock-parallel"
        )

    def subset_output(self, varname: str, output):
        """
        Given an output array, return the slice of the array which we'd
        like to validate against reference data
        """
        return self.tracer_advection.subset_output(varname, output)
