from typing import Optional

from ndsl.constants import Z_DIM
from ndsl.dsl.stencil import StencilFactory
from ndsl.namelist import Namelist

from pace.fv3core.stencils.divergence_damping import DivergenceDamping
from pace.fv3core.testing import TranslateDycoreFortranData2Py


class TranslateDivergenceDamping(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "u": {},
            "v": {},
            "va": {},
            "damped_rel_vort_bgrid": {"serialname": "vort"},
            "ua": {},
            "divg_d": {},
            "vc": {},
            "uc": {},
            "delpc": {},
            "ke": {},
            "rel_vort_agrid": {"serialname": "wk"},
            "nord_col": {},
            "d2_bg": {},
        }
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = {
            "ke": {"iend": grid.ied + 1, "jend": grid.jed + 1},
            "delpc": {},
        }
        self.max_error = 1.4e-10
        self.divdamp: Optional[DivergenceDamping] = None
        self.stencil_factory = stencil_factory
        self.namelist = namelist  # type: ignore

    def compute_from_storage(self, inputs):
        nord_col = self.grid.quantity_factory.zeros(dims=[Z_DIM], units="unknown")
        nord_col.data[:] = nord_col.np.asarray(inputs.pop("nord_col"))
        d2_bg = self.grid.quantity_factory.zeros(dims=[Z_DIM], units="unknown")
        d2_bg.data[:] = d2_bg.np.asarray(inputs.pop("d2_bg"))
        self.divdamp = DivergenceDamping(
            self.stencil_factory,
            self.grid.quantity_factory,
            self.grid.grid_data,
            self.grid.damping_coefficients,
            self.grid.nested,
            self.grid.stretched_grid,
            self.namelist.dddmp,
            self.namelist.d4_bg,
            self.namelist.nord,
            self.namelist.grid_type,
            nord_col,
            d2_bg,
        )
        self.divdamp(**inputs)
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
