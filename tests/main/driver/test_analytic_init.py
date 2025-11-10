from pathlib import Path

import pytest
import yaml

from pace import DriverConfig
from pyfv3 import DynamicalCoreConfig
from pyfv3.initialization.analytic_init import AnalyticCase
from tests.paths import EXAMPLE_CONFIGS_DIR


# TODO: Location of test configurations will be changed after refactor,
#       need to update after

TESTED_CONFIGS: list[Path] = [
    EXAMPLE_CONFIGS_DIR / "analytic_test.yaml",
    EXAMPLE_CONFIGS_DIR / "baroclinic_c48_6ranks_serialbox_test.yaml",
]


@pytest.mark.parametrize(
    "tested_configs",
    [
        pytest.param(TESTED_CONFIGS, id="example configs"),
    ],
)
def test_analytic_init_config(tested_configs: list[Path]):
    for config_file in tested_configs:
        with open(Path(__file__).parent / config_file, "r") as f:
            config = yaml.safe_load(f)
        driver_config = DriverConfig.from_dict(config)
        # Analytic initialization contains a copy of the dynamical core
        # config and analytic case types for addition consistency checks.
        # Other initialization types don't require this.
        if driver_config.initialization.type == "analytic":
            assert (
                type(driver_config.initialization.config.dycore_config)
                == DynamicalCoreConfig
            )
            assert hasattr(driver_config.initialization.config, "case")
            assert type(driver_config.initialization.config.case) == AnalyticCase
        else:
            assert not hasattr(driver_config.initialization, "dycore_config")
