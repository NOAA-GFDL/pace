import math
import os

import numpy as np
import yaml

from pace.driver import Driver, DriverConfig
from pace.util.constants import PI, RADIUS
from pace.util.mpi import MPIComm


DIR = os.path.dirname(os.path.abspath(__file__))


def test_extgrid_equals_generated_1x1():

    with open(
        os.path.join(DIR, "../../driver/examples/configs/test_external_C12_1x1.yaml"),
        "r",
    ) as ext_f:
        ext_config = yaml.safe_load(ext_f)
        ext_driver_config = DriverConfig.from_dict(ext_config)

    ext_driver = Driver(ext_driver_config)

    surface_area_true = 4 * PI * (RADIUS ** 2)

    mpicomm = MPIComm()

    tmp = [math.fsum(row) for row in ext_driver.state.grid_data.area.view[:, :]]
    rank_area_sum = math.fsum(tmp)
    tile_area = mpicomm._comm.gather(rank_area_sum, root=0)

    if mpicomm.Get_rank() == 0:
        total_area = math.fsum(tile_area)
        assert np.isclose(total_area, surface_area_true)
