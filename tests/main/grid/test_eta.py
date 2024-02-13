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


def write_non_mono_eta_file(in_eta_file, out_eta_file):
    """
    Reads in file eta79.nc and alters randomly chosen ak/bk values
    This tests the expected failure of set_eta_hybrid_coefficients
    for coefficients that lead to non-monotonically increasing
    eta values
    """

    data = xr.open_dataset(in_eta_file)
    data["ak"].values[10] = data["ak"].values[0]
    data["bk"].values[20] = 0.0

    data.to_netcdf(out_eta_file)


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


def test_set_hybrid_pressure_coefficients_nofile():

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

    del yaml_config["grid_config"]["config"]["eta_file"]

    try:
        driver_config = pace.driver.DriverConfig.from_dict(yaml_config)
        driver_config.comm_config = pace.driver.NullCommConfig(rank=0, total_ranks=6)
        driver = pace.driver.Driver(config=driver_config)
    except Exception as error:
        if str(error) == "eta file not specified":
            pytest.xfail("testing eta file not specified")
        else:
            pytest.fail(f"ERROR {error}")


def test_set_hybrid_pressure_coefficients_not_mono():

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

    in_eta_file = "tests/main/input/eta79.nc"
    out_eta_file = "eta_not_mono_79.nc"
    write_non_mono_eta_file(in_eta_file, out_eta_file)
    yaml_config["grid_config"]["config"]["eta_file"] = out_eta_file

    try:
        driver_config = pace.driver.DriverConfig.from_dict(yaml_config)
        driver_config.comm_config = pace.driver.NullCommConfig(rank=0, total_ranks=6)
        driver = pace.driver.Driver(config=driver_config)
    except Exception as error:
        if os.path.isfile(out_eta_file):
            os.remove(out_eta_file)
        if str(error) == "ETA values are not monotonically increasing":
            pytest.xfail("testing eta values are not monotomincally increasing")
        else:
            pytest.fail(
                "ERROR in testing etav values not are not monotonically increasing"
            )
