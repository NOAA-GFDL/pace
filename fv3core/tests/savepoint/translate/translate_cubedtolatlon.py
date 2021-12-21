import pace.util as fv3util
from pace.stencils.c2l_ord import CubedToLatLon
from pace.stencils.testing import ParallelTranslate2Py


class TranslateCubedToLatLon(ParallelTranslate2Py):
    inputs = {
        "u": {
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "v": {
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
    }

    def __init__(self, grids, namelist, stencil_factory):
        super().__init__(grids, namelist, stencil_factory)
        grid = grids[0]
        self._base.compute_func = CubedToLatLon(
            stencil_factory, grid.grid_data, order=namelist.c2l_ord
        )
        self._base.in_vars["data_vars"] = {"u": {}, "v": {}, "ua": {}, "va": {}}
        self._base.out_vars = {
            "ua": {},
            "va": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
        }
        self.stencil_factory = stencil_factory
