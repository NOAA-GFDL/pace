import dataclasses
import gc
from typing import Optional

import click
import yaml

from pace.util import AVAILABLE_LOG_LEVELS, pace_log

from .driver import Driver, DriverConfig


@click.command()
@click.argument(
    "CONFIG_PATH",
    required=True,
    type=click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "--log-rank",
    type=click.INT,
    help="rank to log from, or all ranks by default, ignored if running without MPI",
)
@click.option(
    "--log-level",
    default="info",
    help="one of 'debug', 'info', 'warning', 'error', 'critical'",
)
def command_line(config_path: str, log_rank: Optional[int], log_level: str):
    """
    Run the driver.

    CONFIG_PATH is the path to a DriverConfig yaml file.
    """
    level = AVAILABLE_LOG_LEVELS[log_level.lower()]
    pace_log.setLevel(level)
    pace_log.info("loading DriverConfig from yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        driver_config = DriverConfig.from_dict(config)
    pace_log.info(
        f"DriverConfig loaded: {yaml.dump(dataclasses.asdict(driver_config))}"
    )
    main(driver_config=driver_config)


def main(driver_config: DriverConfig) -> Driver:
    driver = Driver(config=driver_config)
    try:
        driver.step_all()
    finally:
        driver.cleanup()
    return driver


if __name__ == "__main__":
    command_line()
    # need to cleanup any python objects that may have MPI operations before
    # mpi4py performs its final cleanup
    gc.collect()
