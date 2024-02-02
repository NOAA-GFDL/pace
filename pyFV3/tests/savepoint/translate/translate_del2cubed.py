from typing import Any, Dict

from ndsl.dsl.stencil import StencilFactory
from ndsl.namelist import Namelist

from pyFV3.stencils.del2cubed import HyperdiffusionDamping
from pyFV3.testing import TranslateDycoreFortranData2Py


class TranslateDel2Cubed(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {"qdel": {}}
        self.in_vars["parameters"] = ["nmax", "cd"]
        self.out_vars: Dict[str, Any] = {"qdel": {}}
        self.stencil_factory = stencil_factory

    def compute_from_storage(self, inputs):
        hyperdiffusion = HyperdiffusionDamping(
            self.stencil_factory,
            quantity_factory=self.grid.quantity_factory,
            damping_coefficients=self.grid.damping_coefficients,
            rarea=self.grid.rarea,
            nmax=inputs.pop("nmax"),
        )
        hyperdiffusion(**inputs)
        return inputs
