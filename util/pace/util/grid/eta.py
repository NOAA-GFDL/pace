from dataclasses import dataclass

import numpy as np
import os

@dataclass
class HybridPressureCoefficients:
    """
    Attributes:
     - ks: The number of pure-pressure layers at the top of the model
        Also the level where model transitions from pure pressure to
        hybrid pressure levels
     - ptop: The pressure at the top of the atmosphere
     - ak: The additive coefficient in the pressure calculation
     - bk: The multiplicative coefficient in the pressure calculation
    """

    ks: int
    ptop: int
    ak: np.ndarray
    bk: np.ndarray


def set_hybrid_pressure_coefficients(km: int) -> HybridPressureCoefficients:
    """
    Sets the coefficients describing the hybrid pressure coordinates.

    The pressure of each k-level is calculated as Pk = ak + (bk * Ps)
    where Ps is the surface pressure. Values are currently stored in
    lookup tables.

    Args:
        km: The number of vertical levels in the model

    Returns:
        a HybridPressureCoefficients dataclass
    """

    # set path where the eta file lives
    GRID_DIR=os.path.join( os.path.abspath('./'), "input/")

    # set filename, e.g, eta79.txt for km=79
    eta_file = GRID_DIR + 'eta' + str(km) + '.txt'
    if( not os.path.isfile(eta_file) ) : raise IOError("file "+eta_file+" does not exist")

    # read file into ak, bk arrays
    ak, bk = np.loadtxt(eta_file, unpack=True)

    # check size of ak and bk array is km+1
    if ak.size-1 != km : raise ValueError("size of ak array is not equal to km="+str(km))
    if bk.size-1 != km : raise ValueError("size of bk array is not equal to km="+str(km))

    # check bk is monotonically increasing
    if( not np.array_equal(bk, bk.sort()) ) : raise ValueError(" bk array is not monotonically increasing")

    if 0.0 in bk:
        ks = 0 if km == 91 else np.where(bk == 0)[0][-1]
        ptop = ak[0]
    else:
        raise ValueError("bk must contain at least one 0.")

    pressure_data = HybridPressureCoefficients(ks, ptop, ak, bk)
    return pressure_data
