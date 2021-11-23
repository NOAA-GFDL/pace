import fv3gfs.util.stencils.updatedzc as updatedzc
from fv3core.testing import TranslateFortranData2Py


class TranslateUpdateDzC(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        update_gz_on_c_grid = updatedzc.UpdateGeopotentialHeightOnCGrid(
            grid.stencil_factory, grid.area
        )

        def compute(**kwargs):
            kwargs["dt"] = kwargs.pop("dt2")
            update_gz_on_c_grid(**kwargs)

        self.compute_func = compute
        self.in_vars["data_vars"] = {
            "dp_ref": {"serialname": "dp0"},
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
