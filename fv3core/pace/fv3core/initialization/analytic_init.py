from enum import Enum

import pace.util as fv3util

from pace.fv3core.dycore_state import DycoreState
from pace.util.grid import GridData


class cases(Enum):
    baroclinic = "baroclinic"
    tropicalcylclone = "tropicalcyclone"


valid_cases = [item.value for item in cases]


def init_analytic_choice(
    analytic_init_str: str,
    grid_data: GridData,
    quantity_factory: fv3util.QuantityFactory,
    adiabatic: bool,
    hydrostatic: bool,
    moist_phys: bool,
    comm: fv3util.CubedSphereCommunicator,
) -> DycoreState:
    """
    This method initializes the choosen analytic test case type
    Args:
        analytic_init_str:      test case specifier
        grid_data:              current selected grid data values
        quantity_factory:       inclusion of QuantityFactory class
        adiabatic:              flag for adiabatic methods
        hydrostatic:            flag for hydrostatic methods
        moist_phys:             flag for including moisture physics methods
        comm:                   inclusion of CubedSphereCommunicator class

    Returns:
        an instance of DycoreState class
    """
    if analytic_init_str in valid_cases:
        if analytic_init_str == "baroclinic":
            import pace.fv3core.initialization.test_cases.initialize_baroclinic as bc

            return bc.init_baroclinic_state(
                grid_data=grid_data,
                quantity_factory=quantity_factory,
                adiabatic=False,
                hydrostatic=False,
                moist_phys=True,
                comm=comm,
            )

        elif analytic_init_str == "tropicalcyclone":
            import pace.fv3core.initialization.test_cases.initialize_tc as tc

            return tc.init_tc_state(
                grid_data=grid_data,
                quantity_factory=quantity_factory,
                hydrostatic=False,
                comm=comm,
            )
        else:
            return None
    else:
        raise ValueError(f"Case {analytic_init_str} not implemented")
