import fv3core.stencils.moist_cv as moist_cv
from fv3core.testing import TranslateFortranData2Py


class TranslateLastStep(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = self.grid.stencil_factory.from_origin_domain(
            moist_cv.moist_pt_last_step,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(add=(0, 0, 1)),
        )
        self.in_vars["data_vars"] = {
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "pt": {},
            "pkz": {"istart": grid.is_, "jstart": grid.js},
            "gz": {
                "serialname": "gz1d",
                "kstart": grid.is_,
                "axis": 0,
                "full_shape": True,
            },
        }
        self.in_vars["parameters"] = ["r_vir", "dtmp"]
        self.out_vars = {
            "gz": {
                "serialname": "gz1d",
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.je,
                "jend": grid.je,
                "kstart": grid.npz - 1,
                "kend": grid.npz - 1,
            },
            "pt": {},
        }
        self.write_vars = ["gz"]
