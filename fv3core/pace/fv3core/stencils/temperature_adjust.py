from gt4py.cartesian.gtscript import PARALLEL, computation, exp, interval, log

import ndsl.util.constants as constants
from ndsl.dsl.typing import Float, FloatField
from pace.fv3core.stencils.basic_operations import sign


def apply_diffusive_heating(
    delp: FloatField,
    delz: FloatField,
    cappa: FloatField,
    heat_source: FloatField,
    pt: FloatField,
    delt_time_factor: Float,
):
    """
    Adjust air temperature from heating due to vorticity damping.
    Heating is limited by deltmax times the length of a timestep, with the
    highest levels limited further.

    Args:
        delp (in): Pressure thickness of atmosphere layers
        delz (in): Vertical thickness of atmosphere layers
        cappa (in): R/Cp
        heat_source (in): heat source from vorticity damping implied by
            energy conservation
        pt (inout): Air potential temperature
        delta_time_factor (in): scaled time step
    """
    with computation(PARALLEL), interval(...):
        pkz = exp(cappa / (1.0 - cappa) * log(constants.RDG * delp / delz * pt))
        dtmp = heat_source / (constants.CV_AIR * delp)
    with computation(PARALLEL):
        with interval(0, 1):
            deltmin = sign(min(delt_time_factor * 0.1, abs(dtmp)), dtmp)
            pt = pt + deltmin / pkz
        with interval(1, 2):
            deltmin = sign(min(delt_time_factor * 0.5, abs(dtmp)), dtmp)
            pt = pt + deltmin / pkz
        with interval(2, None):
            deltmin = sign(min(delt_time_factor, abs(dtmp)), dtmp)
            pt = pt + deltmin / pkz
