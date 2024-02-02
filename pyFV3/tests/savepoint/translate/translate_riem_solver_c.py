from ndsl.dsl.stencil import StencilFactory
from ndsl.namelist import Namelist

from pyFV3.stencils.riem_solver_c import NonhydrostaticVerticalSolverCGrid
from pyFV3.testing import TranslateDycoreFortranData2Py


class TranslateRiem_Solver_C(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.compute_func = NonhydrostaticVerticalSolverCGrid(  # type: ignore
            stencil_factory,
            quantity_factory=self.grid.quantity_factory,
            p_fac=namelist.p_fac,
        )
        self.in_vars["data_vars"] = {
            "cappa": {},
            "hs": {},
            "w3": {},
            "ptc": {},
            "q_con": {},
            "delpc": {},
            "gz": {},
            "pef": {},
            "ws": {},
        }
        self.in_vars["parameters"] = ["dt2", "ptop"]
        self.out_vars = {"pef": {"kend": grid.npz}, "gz": {"kend": grid.npz}}
        self.max_error = 5e-14
        self.stencil_factory = stencil_factory
