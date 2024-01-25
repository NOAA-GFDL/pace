import os
from typing import List

import pytest
import yaml

import pace.driver


DIR = os.path.dirname(os.path.abspath(__file__))

# TODO: Location of test configurations will be changed after refactor,
#       need to update after

TESTED_CONFIGS: List[str] = [
    "../../../driver/examples/configs/analytic_test.yaml",
]


@pytest.mark.parametrize(
    "tested_configs",
    [
        pytest.param(TESTED_CONFIGS, id="example configs"),
    ],
)
def test_analytic_init_config(tested_configs: List[str]):
    for config_file in tested_configs:
        with open(os.path.join(DIR, config_file), "r") as f:
            config = yaml.safe_load(f)
        driver_config = pace.driver.DriverConfig.from_dict(config)
        assert driver_config.initialization.type == "analytic"
