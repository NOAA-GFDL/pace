import fv3core._config as spec
from fv3core.testing import TranslateFortranData2Py
from fv3gfs.util.stencils.divergence_damping import DivergenceDamping


class TranslateA2B_Ord4(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"wk": {}, "vort": {}, "delpc": {}, "nord_col": {}}
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = {"wk": {}, "vort": {}}

    def compute_from_storage(self, inputs):
        divdamp = DivergenceDamping(
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
            inputs["nord_col"],
        )
        del inputs["nord_col"]
        divdamp._vorticity_calc(**inputs)
        return inputs
