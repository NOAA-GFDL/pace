from dataclasses import dataclass

import numpy as np


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

    #set filename, ex, eta79.txt; read file into ak, bk arrays
    etafile = 'eta ' + str(km) + 'txt'
    ak, bk = np.loadtxt(etafile, dtype=np.float(64), unpack=True)

    # check size of ak and bk array is km
    if ak.size != km : raise ValueError("size of ak array is not equal to "+str(km))
    if bk.size != km : raise ValueError("size of bk array is not equal to "+str(km))

    if 0.0 in bk:
        ks = 0 if km == 91 else np.where(bk == 0)[0][-1]
        ptop = ak[0]
    else:
        raise ValueError("bk must contain at least one 0.")

    pressure_data = HybridPressureCoefficients(ks, ptop, ak, bk)
    return pressure_data
