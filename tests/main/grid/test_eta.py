#!/usr/bin/env python3

import numpy as np
import pytest

from pace.util.grid.eta import set_hybrid_pressure_coefficients


km = 79
eta_file = "./input/eta79.txt"
ak_79, bk_79 = np.loadtxt(eta_file, unpack=True)


@pytest.mark.xfail
def test_set_hybrid_pressure_coefficients_nofile():

    """ File does not exist.  Test should fail """

    pressure_data = set_hybrid_pressure_coefficients(km=90)


def test_set_hybrid_pressure_coefficients_correct():

    """  Good values of ak, bk.  Test should pass """

    ks = 18
    ptop = 300.0

    pressure_data = set_hybrid_pressure_coefficients(km)

    if not np.array_equal(ak_79, pressure_data.ak):
        raise ValueError("Unexpected values in ak array")

    if not np.array_equal(bk_79, pressure_data.bk):
        raise ValueError("Unexpected values in bk array")

    if ks != pressure_data.ks:
        raise ValueError("Unexpected ks value")

    if ptop != pressure_data.ptop:
        raise ValueError("Unexpected ptopt value")


@pytest.mark.xfail
def test_set_hybrid_pressure_coefficients_nonincreasing():

    """
    Array bk is not monotonically increasing.
    Test is expected to fail
    """

    ak_cp = ak_79[:]
    bk_cp = bk_79[:]

    bk_cp[10] = 0.1  # random number
    ak_cp[13] = 1.4  # random number

    pressure_data = set_hybrid_pressure_coefficients(km)
