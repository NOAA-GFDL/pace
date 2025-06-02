from pathlib import Path
from typing import List

import pytest
import yaml

from pace import DriverConfig
from tests.paths import EXAMPLE_CONFIGS_DIR


# TODO: Location of test configurations will be changed after refactor,
#       need to update after

TESTED_CONFIGS: List[Path] = [
    EXAMPLE_CONFIGS_DIR / "analytic_test.yaml",
]


@pytest.mark.parametrize(
    "tested_configs",
    [
        pytest.param(TESTED_CONFIGS, id="example configs"),
    ],
)
def test_analytic_init_config(tested_configs: List[Path]):
    for config_file in tested_configs:
        with open(Path(__file__).parent / config_file, "r") as f:
            config = yaml.safe_load(f)
        driver_config = DriverConfig.from_dict(config)
        assert driver_config.initialization.type == "analytic"
