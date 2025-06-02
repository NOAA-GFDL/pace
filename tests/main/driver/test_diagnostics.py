from pathlib import Path

import xarray as xr
import yaml

from pace import DiagnosticsConfig, DriverConfig, NullCommConfig
from pace.run import main
from tests.paths import EXAMPLE_CONFIGS_DIR


def test_diagnostics_can_be_opened(tmpdir: Path):
    with open(EXAMPLE_CONFIGS_DIR / "baroclinic_c12.yaml", "r") as f:
        driver_config = DriverConfig.from_dict(yaml.safe_load(f))
    diagnostics_path = tmpdir / "output.zarr"
    driver_config.diagnostics_config = DiagnosticsConfig(
        path=diagnostics_path,
        names=["u", "v", "ua", "va", "w", "delp", "pt", "qvapor"],
    )
    driver_config.comm_config = NullCommConfig(rank=0, total_ranks=6)
    driver_config.dt_atmos = 60
    driver_config.minutes = 1
    main(driver_config)
    xr.open_zarr(diagnostics_path)
