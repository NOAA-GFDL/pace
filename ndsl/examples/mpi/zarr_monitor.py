from datetime import timedelta

import cftime
import numpy as np
import zarr
from mpi4py import MPI

import ndsl.util


OUTPUT_PATH = "output/zarr_monitor.zarr"


def get_example_state(time):
    sizer = ndsl.util.SubtileGridSizer(
        nx=48, ny=48, nz=70, n_halo=3, extra_dim_lengths={}
    )
    allocator = ndsl.util.QuantityFactory(sizer, np)
    air_temperature = allocator.zeros(
        [ndsl.util.X_DIM, ndsl.util.Y_DIM, ndsl.util.Z_DIM], units="degK"
    )
    air_temperature.view[:] = np.random.randn(*air_temperature.extent)
    return {"time": time, "air_temperature": air_temperature}


if __name__ == "__main__":
    size = MPI.COMM_WORLD.Get_size()
    # assume square tile faces
    ranks_per_edge = int((size // 6) ** 0.5)
    layout = (ranks_per_edge, ranks_per_edge)

    store = zarr.storage.DirectoryStore(OUTPUT_PATH)
    partitioner = ndsl.util.CubedSpherePartitioner(ndsl.util.TilePartitioner(layout))
    monitor = ndsl.util.ZarrMonitor(store, partitioner, mpi_comm=MPI.COMM_WORLD)

    time = cftime.DatetimeJulian(2020, 1, 1)
    timestep = timedelta(hours=1)

    for i in range(10):
        state = get_example_state(time)
        monitor.store(state)
        time += timestep
