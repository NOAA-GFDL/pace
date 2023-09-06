from typing import List

import pytest
import yaml

import pace.driver


TESTED_CONFIGS: List[str] = [
    "../../../driver/examples/configs/analytic_test.yaml",
]


@pytest.mark.parametrize(
    "tested_configs"[
        pytest.param(TESTED_CONFIGS, id="example configs"),
    ],
)
def test_analytic_init_config(tested_configs: List[str]):
    for config_file in tested_configs:
        config = yaml.safe_load(config_file)
        driver_config = pace.driver.DriverConfig.from_dict(config)
        driver = pace.driver.Driver(config=driver_config)
