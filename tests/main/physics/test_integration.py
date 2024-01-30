from dataclasses import fields
from datetime import timedelta

import numpy as np

import ndsl.dsl
import pace.physics
import ndsl.util
import ndsl.util.grid
from ndsl.dsl.stencil_config import CompilationConfig
from ndsl.stencils.testing import assert_same_temporaries, copy_temporaries


try:
    import cupy as cp
except ImportError:
    cp = np


def setup_physics():
    backend = "numpy"
    layout = (1, 1)
    physics_config = pace.physics.PhysicsConfig(
        dt_atmos=225, hydrostatic=False, npx=13, npy=13, npz=79, nwat=6, do_qa=True
    )
    mpi_comm = ndsl.util.NullComm(
        rank=0, total_ranks=6 * layout[0] * layout[1], fill_value=0.0
    )
    partitioner = ndsl.util.CubedSpherePartitioner(ndsl.util.TilePartitioner(layout))
    communicator = ndsl.util.CubedSphereCommunicator(mpi_comm, partitioner)
    sizer = ndsl.util.SubtileGridSizer.from_tile_params(
        nx_tile=physics_config.npx - 1,
        ny_tile=physics_config.npy - 1,
        nz=physics_config.npz,
        n_halo=3,
        extra_dim_lengths={},
        layout=layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
    )
    grid_indexing = ndsl.dsl.stencil.GridIndexing.from_sizer_and_communicator(
        sizer=sizer, comm=communicator
    )
    quantity_factory = ndsl.util.QuantityFactory.from_backend(
        sizer=sizer, backend=backend
    )
    dace_config = ndsl.dsl.DaceConfig(
        communicator=communicator,
        backend=backend,
        orchestration=ndsl.dsl.DaCeOrchestration.Python,
    )
    stencil_config = ndsl.dsl.stencil.StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend,
            rebuild=False,
            validate_args=True,
        ),
        dace_config=dace_config,
    )
    stencil_factory = ndsl.dsl.stencil.StencilFactory(
        config=stencil_config,
        grid_indexing=grid_indexing,
    )
    metric_terms = ndsl.util.grid.MetricTerms(
        quantity_factory=quantity_factory,
        communicator=communicator,
        eta_file="tests/main/input/eta79.nc",
    )
    grid_data = ndsl.util.grid.GridData.new_from_metric_terms(metric_terms)
    physics = pace.physics.Physics(
        stencil_factory,
        quantity_factory,
        grid_data,
        physics_config,
    )
    physics_state = pace.physics.PhysicsState.init_zeros(
        quantity_factory, schemes=[pace.physics.PHYSICS_PACKAGES["GFS_microphysics"]]
    )
    random = np.random.RandomState(0)
    for field in fields(pace.physics.PhysicsState):
        array = getattr(physics_state, field.name)
        # check that it's a storage this way, because Field is not a class
        if isinstance(array, (np.ndarray, cp.ndarray)):
            array[:] = random.uniform(-1, 1, size=array.data.shape)
    return physics, physics_state


def test_temporaries_are_deterministic():
    """
    This is a precursor test to the next one, ensuring that two
    identically-initialized dycores called on identically-initialized
    states produce identical temporaries.

    This will fail if there is non-determinism in the initialization,
    for example from using `empty` instead of `zeros` to initialize data.
    """
    physics1, state_1 = setup_physics()
    physics2, state_2 = setup_physics()

    physics1(state_1, timestep=timedelta(minutes=5).total_seconds())
    first_temporaries = copy_temporaries(physics1, max_depth=10)
    assert len(first_temporaries) > 0
    physics2(state_2, timestep=timedelta(minutes=5).total_seconds())
    second_temporaries = copy_temporaries(physics2, max_depth=10)
    assert_same_temporaries(second_temporaries, first_temporaries)


def test_call_on_same_state_same_physics_produces_same_temporaries():
    """
    Assuming the precursor test passes, this test indicates whether
    the physics retains and re-uses internal state on subsequent calls.
    If it does not, then subsequent calls on identical input should
    produce identical results.
    """
    physics, state_1 = setup_physics()
    _, state_2 = setup_physics()

    # state_1 and state_2 are identical, if the physics is stateless then they
    # should produce identical physics final states when used to call
    physics(state_1, timestep=timedelta(minutes=5).total_seconds())
    first_temporaries = copy_temporaries(physics, max_depth=10)
    assert len(first_temporaries) > 0
    physics(state_2, timestep=timedelta(minutes=5).total_seconds())
    second_temporaries = copy_temporaries(physics, max_depth=10)
    assert_same_temporaries(second_temporaries, first_temporaries)
