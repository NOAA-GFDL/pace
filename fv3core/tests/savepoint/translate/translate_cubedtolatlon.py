import ndsl.dsl
import ndsl.util
from ndsl.stencils.c2l_ord import CubedToLatLon
from ndsl.stencils.testing import ParallelTranslate2Py


class TranslateCubedToLatLon(ParallelTranslate2Py):
    inputs = {
        "u": {
            "dims": [ndsl.util.X_DIM, ndsl.util.Y_INTERFACE_DIM, ndsl.util.Z_DIM],
            "units": "m/s",
        },
        "v": {
            "dims": [ndsl.util.X_INTERFACE_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM],
            "units": "m/s",
        },
    }

    def __init__(
        self,
        grid,
        namelist: ndsl.util.Namelist,
        stencil_factory: ndsl.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self._base.in_vars["data_vars"] = {"u": {}, "v": {}, "ua": {}, "va": {}}
        self._base.out_vars = {
            "ua": {},
            "va": {},
            "u": self.grid.y3d_domain_dict(),
            "v": self.grid.x3d_domain_dict(),
        }
        self.stencil_factory = stencil_factory
        self.grid_type = namelist.grid_type

    def compute_parallel(self, inputs, communicator):
        self._base.make_storage_data_input_vars(inputs)
        u_quantity = _quantity_wrap(
            inputs["u"],
            self.inputs["u"]["dims"],
            self.grid.grid_indexing,
        )
        v_quantity = _quantity_wrap(
            inputs["v"],
            self.inputs["v"]["dims"],
            self.grid.grid_indexing,
        )
        state_dict = {"u": u_quantity, "v": v_quantity}

        self._cubed_to_latlon = CubedToLatLon(
            state=state_dict,
            stencil_factory=self.stencil_factory,
            quantity_factory=self.grid.quantity_factory,
            grid_data=self.grid.grid_data,
            order=self.namelist.c2l_ord,
            comm=communicator,
            grid_type=self.grid_type,
        )
        self._cubed_to_latlon(**inputs)
        return self._base.slice_output(inputs)


def _quantity_wrap(storage, dims, grid_indexing):
    origin, extent = grid_indexing.get_origin_domain(dims)
    return ndsl.util.Quantity(
        storage,
        dims=dims,
        units="unknown",
        origin=origin,
        extent=extent,
    )
