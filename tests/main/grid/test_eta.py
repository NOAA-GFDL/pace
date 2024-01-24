#!/usr/bin/env python3

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


def set_answers(config_file):

    """
    Read in the expected values of ak and bk
    arrays from the input eta netcdf files.
    """

    if "79" in config_file:
        eta_file = "tests/main/input/eta79.nc"
    if "91" in config_file:
        eta_file = "tests/main/input/eta91.nc"
    data = xr.open_dataset(eta_file)
    return data["ak"].values, data["bk"].values


@pytest.mark.parametrize("km", [79, 91])
def test_set_hybrid_pressure_coefficients_correct(km):

    """This test checks to see if the ak and bk arrays
    are read-in correctly are stored as
    expected.  Both values of km=79 and km=91 are
    tested.
    """

    config_file = f"tests/main/grid/input/test_config_{km}.yaml"
    with open(config_file, "r") as f:
        yaml_config = yaml.safe_load(f)

    driver_config = pace.driver.DriverConfig.from_dict(yaml_config)
    driver_config.comm_config = pace.driver.NullCommConfig(rank=0, total_ranks=6)
    driver = pace.driver.Driver(config=driver_config)

    p_results = driver.state.grid_data.p.data
    ak_results = driver.state.grid_data.ak.data
    bk_results = driver.state.grid_data.bk.data
    ak_answers, bk_answers = set_answers(config_file)

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
        "tests/main/grid/input/test_config_nofile.yaml",
        "tests/main/grid/input/test_config_not_mono.yaml",
    ],
)
@pytest.mark.xfail
def test_set_hybrid_pressure_coefficients_fail(cfile):

    """This test checks to see that the program
    fails when (1) the eta_file is not specified in the yaml
    configuration file; and (2), the computed eta values
    increase non-monotonically.  For the latter test, the eta_file
    is specified in test_config_not_mono.yaml file and
    the ak and bk values in the eta_file have been nonsensically changed
    to result in erronenous eta values.
    """

    config_file = cfile
    with open(config_file, "r") as f:
        yaml_config = yaml.safe_load(f)

    driver_config = pace.driver.DriverConfig.from_dict(yaml_config)
    driver_config.comm_config = pace.driver.NullCommConfig(rank=0, total_ranks=6)
    driver = pace.driver.Driver(config=driver_config)
