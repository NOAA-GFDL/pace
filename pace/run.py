import dataclasses
import gc

import click
import yaml

from ndsl.logging import AVAILABLE_LOG_LEVELS, ndsl_log
from pace.driver import Driver, DriverConfig


@click.command()
@click.argument(
    "CONFIG_PATH",
    required=True,
    type=click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "--log-level",
    default="info",
    help="one of 'debug', 'info', 'warning', 'error', 'critical'",
)
def command_line(config_path: str, log_level: str) -> None:
    """
    Run the driver.

    CONFIG_PATH is the path to a DriverConfig yaml file.
    """
    level = AVAILABLE_LOG_LEVELS[log_level.lower()]
    ndsl_log.setLevel(level)
    ndsl_log.info("loading DriverConfig from yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        driver_config = DriverConfig.from_dict(config)
    ndsl_log.info(
        f"DriverConfig loaded: {yaml.dump(dataclasses.asdict(driver_config))}"
    )
    main(driver_config=driver_config)


def main(driver_config: DriverConfig) -> None:
    driver = Driver(config=driver_config)
    try:
        driver.step_all()
    finally:
        driver.cleanup()


if __name__ == "__main__":
    command_line()
    # need to cleanup any python objects that may have MPI operations before
    # mpi4py performs its final cleanup
    gc.collect()
