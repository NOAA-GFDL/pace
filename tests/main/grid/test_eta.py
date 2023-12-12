#!/usr/bin/env python3

import numpy as np
import pytest
import xarray as xr
import yaml

import pace.driver


def set_answers(config_file):

    eta_file = "./input/eta79.nc"
    data = xr.open_dataset(eta_file)
    return data["ak"].values, data["bk"].values


def test_set_hybrid_pressure_coefficients_correct():

    config_file = "./input/test_config_79.yaml"
    with open(config_file, "r") as f:
        yaml_config = yaml.safe_load(f)

    driver_config = pace.driver.DriverConfig.from_dict(yaml_config)
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


@pytest.mark.xfail
def test_set_hybrid_pressure_coefficients_nofile():

    config_file = "./input/test_config_nofile.yaml"
    with open(config_file, "r") as f:
        yaml_config = yaml.safe_load(f)

    driver_config = pace.driver.DriverConfig.from_dict(yaml_config)
    driver = pace.driver.Driver(config=driver_config)


@pytest.mark.xfail
def test_set_hybrid_pressure_coefficients_nonmonotonic():

    config_file = "./input/test_config_not_mono.yaml"
    with open(config_file, "r") as f:
        yaml_config = yaml.safe_load(f)

    driver_config = pace.driver.DriverConfig.from_dict(yaml_config)
    driver = pace.driver.Driver(config=driver_config)