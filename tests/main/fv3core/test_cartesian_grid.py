import numpy as np
import pytest

import pace.util
from pace.util.constants import PI
from pace.util.grid.generation import MetricTerms


@pytest.mark.parametrize("npx", [8])
@pytest.mark.parametrize("npy", [8])
@pytest.mark.parametrize("npz", [1])
@pytest.mark.parametrize("dx_const", [1e2, 1e3])
@pytest.mark.parametrize("dy_const", [2e2, 3e3])
@pytest.mark.parametrize("deglat", [0.0, 15.0])
@pytest.mark.parametrize("backend", ["numpy"])
def test_cartesian_grid_generation(
    npx: int,
    npy: int,
    npz: int,
    dx_const: float,
    dy_const: float,
    deglat: float,
    backend: str,
):
    mpi_comm = pace.util.NullComm(rank=0, total_ranks=1)
    partitioner = pace.util.TilePartitioner((1, 1))
    communicator = pace.util.TileCommunicator(mpi_comm, partitioner)
    grid_generator = MetricTerms.from_tile_sizing(
        npx=npx,
        npy=npy,
        npz=npz,
        communicator=communicator,
        backend=backend,
        grid_type=4,
        dx_const=dx_const,
        dy_const=dy_const,
        deglat=deglat,
    )
    assert np.all(grid_generator.lat_agrid.data == deglat * PI / 180.0)
    assert np.all(grid_generator.lon_agrid.data == 0.0)
    for prop in ("dx", "dxa", "dxc"):
        dx = getattr(grid_generator, prop)
        assert np.all(dx.data == dx_const)
    for prop in ("dy", "dya", "dyc"):
        dy = getattr(grid_generator, prop)
        assert np.all(dy.data == dy_const)
    for prop in ("rdx", "rdxa", "rdxc"):
        rdx = getattr(grid_generator, prop)
        assert np.all(rdx.data == 1.0 / dx_const)
    for prop in ("rdy", "rdya", "rdyc"):
        rdy = getattr(grid_generator, prop)
        assert np.all(rdy.data == 1.0 / dy_const)
    for prop in ("area", "area_c"):
        area = getattr(grid_generator, prop)
        assert np.all(area.data == dx_const * dy_const)
    for prop in ("rarea", "rarea_c"):
        rarea = getattr(grid_generator, prop)
        assert np.all(rarea.data == 1.0 / (dx_const * dy_const))
    for prop in ("ec1", "ew1", "es1"):
        unit_x = getattr(grid_generator, prop)
        assert np.all(unit_x.data[..., 0] == 1.0)
        assert np.all(unit_x.data[..., 1:] == 0.0)
    for prop in ("ec2", "ew2", "es2"):
        unit_y = getattr(grid_generator, prop)
        assert np.all(unit_y.data[..., 0] == 0.0)
        assert np.all(unit_y.data[..., 1] == 1.0)
        assert np.all(unit_y.data[..., 2] == 0.0)
    for i in range(1, 10):
        cos_sg = getattr(grid_generator, f"cos_sg{i}")
        assert np.all(cos_sg.data == 0.0)
        sin_sg = getattr(grid_generator, f"sin_sg{i}")
        assert np.all(sin_sg.data == 1.0)
    for prop in ("cosa", "cosa_u", "cosa_v", "cosa_s"):
        cos = getattr(grid_generator, prop)
        assert np.all(cos.data == 0.0)
    for prop in ("sina", "sina_u", "sina_v", "rsina", "rsin_u", "rsin_v", "rsin2"):
        sin = getattr(grid_generator, prop)
        assert np.all(sin.data == 1.0)
