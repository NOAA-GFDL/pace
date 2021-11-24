from typing import Optional

import fv3core._config as spec
import fv3core.stencils.divergence_damping
from fv3core.testing import TranslateFortranData2Py


class TranslateDivergenceDamping(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "u": {},
            "v": {},
            "va": {},
            "u_contra_dyc": {"serialname": "ptc"},
            "v_contra_dxc": {"serialname": "vort"},
            "ua": {},
            "divg_d": {},
            "vc": {},
            "uc": {},
            "delpc": {},
            "ke": {},
            "wk": {},
            "nord_col": {},
            "d2_bg": {},
        }
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = {
            "v_contra_dxc": {"serialname": "vort"},
            "ke": {"iend": grid.ied + 1, "jend": grid.jed + 1},
            "delpc": {},
        }
        self.max_error = 1.4e-10
        self.divdamp: Optional[
            fv3core.stencils.divergence_damping.DivergenceDamping
        ] = None

    def compute_from_storage(self, inputs):
        self.divdamp = fv3core.stencils.divergence_damping.DivergenceDamping(
            self.grid.stencil_factory,
            self.grid.grid_data,
            self.grid.damping_coefficients,
            self.grid.nested,
            self.grid.stretched_grid,
            spec.namelist.dddmp,
            spec.namelist.d4_bg,
            spec.namelist.nord,
            spec.namelist.grid_type,
            inputs["nord_col"],
            inputs["d2_bg"],
        )
        del inputs["nord_col"]
        del inputs["d2_bg"]
        self.divdamp(**inputs)
        inputs["v_contra_dxc"] = self.subset_output(
            "v_contra_dxc", inputs["v_contra_dxc"]
        )
        return inputs

    def subset_output(self, varname: str, output):
        """
        Given an output array, return the slice of the array which we'd
        like to validate against reference data
        """
        if self.divdamp is None:
            raise RuntimeError("must call compute_from_storage before subset_output")
        else:
            return self.divdamp.subset_output(varname, output)  # type: ignore
