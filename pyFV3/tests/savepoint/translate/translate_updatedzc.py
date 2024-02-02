import numpy as np
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.stencil import StencilFactory
from ndsl.namelist import Namelist

import pyFV3.stencils.updatedzc as updatedzc
from pyFV3.testing import TranslateDycoreFortranData2Py
from pyFV3.utils.functional_validation import get_subset_func


class TranslateUpdateDzC(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.stencil_factory = stencil_factory
        update_gz_on_c_grid = updatedzc.UpdateGeopotentialHeightOnCGrid(
            self.stencil_factory,
            quantity_factory=self.grid.quantity_factory,
            area=grid.grid_data.area,
            dp_ref=grid.grid_data.dp_ref,
            grid_type=namelist.grid_type,
        )

        def compute(**kwargs):
            kwargs["dt"] = kwargs.pop("dt2")
            update_gz_on_c_grid(**kwargs)

        self.compute_func = compute  # type: ignore
        self.in_vars["data_vars"] = {
            "zs": {},
            "ut": {"serialname": "utc"},
            "vt": {"serialname": "vtc"},
            "gz": {},
            "ws": {},
        }
        self.in_vars["parameters"] = ["dt2"]
        self.out_vars = {
            "gz": grid.default_buffer_k_dict(),
            "ws": {"kstart": -1, "kend": None},
        }
        self._subset = get_subset_func(
            self.grid.grid_indexing,
            dims=[X_DIM, Y_DIM, Z_DIM],
            n_halo=((0, 0), (0, 0)),
        )
        self._subset_2d = get_subset_func(
            self.grid.grid_indexing,
            dims=[X_DIM, Y_DIM],
            n_halo=((0, 0), (0, 0)),
        )

    def compute(self, inputs):
        outputs = super().compute(inputs)
        outputs["gz"] = self.subset_output("gz", outputs["gz"])
        outputs["ws"] = self.subset_output("ws", outputs["ws"])
        return outputs

    def subset_output(self, varname: str, output: np.ndarray) -> np.ndarray:
        """
        Given an output array, return the slice of the array which we'd
        like to validate against reference data
        """
        if varname in ["gz"]:
            return self._subset(output)
        elif varname in ["ws"]:
            return self._subset_2d(output)
        else:
            return output
