import os

import xarray as xr
import yaml

from pace.driver import DiagnosticsConfig, DriverConfig, NullCommConfig
from pace.driver.run import main


DIR = os.path.dirname(os.path.abspath(__file__))


def test_diagnostics_can_be_opened(tmpdir):
    with open(
        os.path.join(DIR, "../../../driver/examples/configs/baroclinic_c12.yaml"), "r"
    ) as f:
        driver_config = DriverConfig.from_dict(yaml.safe_load(f))
    diagnostics_path = os.path.join(tmpdir, "output.zarr")
    driver_config.diagnostics_config = DiagnosticsConfig(
        path=diagnostics_path,
        names=["u", "v", "ua", "va", "w", "delp", "pt", "qvapor"],
    )
    driver_config.comm_config = NullCommConfig(rank=0, total_ranks=6)
    driver_config.dt_atmos = 60
    driver_config.minutes = 1
    main(driver_config)
    xr.open_zarr(diagnostics_path)
