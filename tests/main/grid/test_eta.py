#!/usr/bin/env python3

import os

import numpy as np
import pytest
import xarray as xr
import yaml

import pace.driver


"""
This test checks to ensure that ak and bk
values are read-in and stored properly.
In addition, this test checks to ensure that
the function set_hybrid_pressure_coefficients
fail as expected if the computed eta values
vary non-mononitically and if the eta_file
is not provided.
"""


def set_answers(eta_file):

    """
    Read in the expected values of ak and bk
    arrays from the input eta NetCDF files.
    """

    data = xr.open_dataset(eta_file)
    return data["ak"].values, data["bk"].values


@pytest.mark.parametrize("km", [79, 91])
def test_set_hybrid_pressure_coefficients_correct(km):

    """This test checks to see that the ak and bk arrays
    are read-in correctly and are stored as
    expected.  Both values of km=79 and km=91 are
    tested and both tests are expected to pass
    with the stored ak and bk values agreeing with the
    values read-in directly from the NetCDF file.
    """

    dirname = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(
        dirname, "../../../driver/examples/configs/baroclinic_c12.yaml"
    )

    with open(config_file, "r") as f:
        yaml_config = yaml.safe_load(f)

    yaml_config["nz"] = km
    yaml_config["grid_config"]["config"]["eta_file"] = f"tests/main/input/eta{km}.nc"

    driver_config = pace.driver.DriverConfig.from_dict(yaml_config)
    driver_config.comm_config = pace.driver.NullCommConfig(rank=0, total_ranks=6)
    driver = pace.driver.Driver(config=driver_config)

    p_results = driver.state.grid_data.p.data
    ak_results = driver.state.grid_data.ak.data
    bk_results = driver.state.grid_data.bk.data
    ak_answers, bk_answers = set_answers(f"tests/main/input/eta{km}.nc")

    if ak_answers.size != ak_results.size:
        raise ValueError("Unexpected size of bk")
    if bk_answers.size != bk_results.size:
        raise ValueError("Unexpected size of ak")

    if not np.array_equal(ak_answers, ak_results):
        raise ValueError("Unexpected value of ak")
    if not np.array_equal(bk_answers, bk_results):
        raise ValueError("Unexpected value of bk")

    driver.safety_checker.clear_all_checks()


@pytest.mark.parametrize(
    "cfile",
    [
        "file_is_not_here",
        "tests/main/grid/input/eta_not_mono.nc",
    ],
)
@pytest.mark.xfail
def test_set_hybrid_pressure_coefficients_fail(cfile):

    """This test checks to see that the program
    fails when (1) the eta_file is not specified in the yaml
    configuration file; and (2), the computed eta values
    increase non-monotonically.  For the latter test, the eta_file
    is specified in test_config_not_mono.yaml file and
    the ak and bk values in the eta_file have been changed nonsensically
    to result in erronenous eta values.
    """

    dirname = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(
        dirname, "../../../driver/examples/configs/baroclinic_c12.yaml"
    )

    with open(config_file, "r") as f:
        yaml_config = yaml.safe_load(f)

    yaml_config["grid_config"]["config"]["eta_file"] = cfile

    driver_config = pace.driver.DriverConfig.from_dict(yaml_config)
    driver_config.comm_config = pace.driver.NullCommConfig(rank=0, total_ranks=6)
    driver = pace.driver.Driver(config=driver_config)
