import os
from dataclasses import dataclass

import numpy as np
import xrray as xr

from .utils import compute_eta


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


def set_hybrid_pressure_coefficients(
    km: int, eta_file: str
) -> HybridPressureCoefficients:
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

    if eta_file == "None":
        raise IOError("eta file not specified")
    if not os.path.isfile(eta_file):
        raise IOError("file " + eta_file + " does not exist")

    # read file into ak, bk arrays
    data = xr.open_dataset(eta_file)
    ak = data["ak"].values
    bk = data["bk"].values

    # check size of ak and bk array is km+1
    if ak.size - 1 != km:
        raise ValueError("size of ak array is not equal to km=" + str(km))
    if bk.size - 1 != km:
        raise ValueError("size of bk array is not equal to km=" + str(km))

    # check that the eta values computed from ak and bk are monotonically increasing
    eta, etav = compute_eta(ak, bk)

    (eta_sorted, etav_sorted) = (np.sort(eta), np.sort(etav))
    for i in range(eta.size):
        if eta_sorted[i] != eta[i]:
            raise ValueError("ETA values are not monotonically increasing")
        if etav_sorted[i] != etav[i]:
            raise ValueError("ETAV values are not monotonically increasing")

    if 0.0 in bk:
        ks = 0 if km == 91 else np.where(bk == 0)[0][-1]
        ptop = ak[0]
    else:
        raise ValueError("bk must contain at least one 0.")

    pressure_data = HybridPressureCoefficients(ks, ptop, ak, bk)

    return pressure_data
