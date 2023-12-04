from enum import Enum

import pace.util as fv3util
from pace.util import MetaEnumStr
from pace.fv3core.dycore_state import DycoreState
from pace.util.grid import GridData


class Cases(Enum, metaclass=MetaEnumStr):
    baroclinic = "baroclinic"
    tropicalcyclone = "tropicalcyclone"


def init_analytic_state(
    analytic_init_case: str,
    grid_data: GridData,
    quantity_factory: fv3util.QuantityFactory,
    adiabatic: bool,
    hydrostatic: bool,
    moist_phys: bool,
    comm: fv3util.Communicator,
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
    if analytic_init_case in Cases:
        if analytic_init_case == Cases.baroclinic.value:
            import pace.fv3core.initialization.test_cases.initialize_baroclinic as bc

            assert isinstance(comm, fv3util.CubedSphereCommunicator)

            return bc.init_baroclinic_state(
                grid_data=grid_data,
                quantity_factory=quantity_factory,
                adiabatic=adiabatic,
                hydrostatic=hydrostatic,
                moist_phys=moist_phys,
                comm=comm,
            )

        elif analytic_init_case == Cases.tropicalcyclone.value:
            import pace.fv3core.initialization.test_cases.initialize_tc as tc

            assert isinstance(comm, fv3util.CubedSphereCommunicator)

            return tc.init_tc_state(
                grid_data=grid_data,
                quantity_factory=quantity_factory,
                hydrostatic=hydrostatic,
                comm=comm,
            )
        else:
            raise ValueError(f"Case {analytic_init_case} not implemented")
    else:
        raise ValueError(f"Case {analytic_init_case} not recognized")
