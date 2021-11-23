import fv3core._config as spec
from fv3core.testing import TranslateFortranData2Py
from fv3gfs.util.stencils.d2a2c_vect import DGrid2AGrid2CGridVectors


class TranslateD2A2C_Vect(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        dord4 = True
        self.compute_func = DGrid2AGrid2CGridVectors(
            self.grid.stencil_factory,
            self.grid.grid_data,
            self.grid.nested,
            spec.namelist.grid_type,
            dord4,
        )
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "u": {},
            "v": {},
            "ua": {},
            "va": {},
            "utc": {},
            "vtc": {},
        }
        self.out_vars = {
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "ua": {},
            "va": {},
            "utc": {},
            "vtc": {},
        }
        # TODO: This seems to be needed primarily for the edge_interpolate_4
        # methods, can we rejigger the order of operations to make it match to
        # more precision?
        self.max_error = 2e-10
