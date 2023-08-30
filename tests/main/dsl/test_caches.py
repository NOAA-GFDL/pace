import pytest
from gt4py.cartesian.gtscript import PARALLEL, Field, computation, interval
from gt4py.storage import empty, ones

import pace.dsl
from pace.dsl.dace import orchestrate
from pace.dsl.dace.dace_config import DaceConfig, DaCeOrchestration
from pace.dsl.stencil import CompilationConfig, GridIndexing


def _make_storage(
    func,
    grid_indexing,
    stencil_config: pace.dsl.StencilConfig,
    *,
    dtype=float,
    aligned_index=(0, 0, 0),
):
    return func(
        backend=stencil_config.compilation_config.backend,
        shape=grid_indexing.domain,
        dtype=dtype,
        aligned_index=aligned_index,
    )


def _stencil(inp: Field[float], out: Field[float], scalar: float):
    with computation(PARALLEL), interval(...):
        out = inp


def _build_stencil(backend, orchestrated: DaCeOrchestration):
    # Make stencil and verify it ran
    grid_indexing = GridIndexing(
        domain=(5, 5, 5),
        n_halo=2,
        south_edge=True,
        north_edge=True,
        west_edge=True,
        east_edge=True,
    )

    stencil_config = pace.dsl.StencilConfig(
        compilation_config=CompilationConfig(backend=backend, rebuild=True),
        dace_config=DaceConfig(None, backend, 5, 5, orchestrated),
    )

    stencil_factory = pace.dsl.StencilFactory(stencil_config, grid_indexing)

    built_stencil = stencil_factory.from_origin_domain(
        _stencil, (0, 0, 0), domain=grid_indexing.domain
    )

    return built_stencil, grid_indexing, stencil_config


class OrchestratedProgam:
    def __init__(self, backend, orchestration):
        self.stencil, grid_indexing, stencil_config = _build_stencil(
            backend, orchestration
        )
        orchestrate(obj=self, config=stencil_config.dace_config)
        self.inp = _make_storage(ones, grid_indexing, stencil_config, dtype=float)
        self.out = _make_storage(empty, grid_indexing, stencil_config, dtype=float)

    def __call__(self):
        self.stencil(self.inp, self.out, self.inp[0, 0, 0])


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("dace:cpu"),
    ],
)
def test_relocatability_orchestration(backend):
    import os
    import shutil

    from gt4py.cartesian import config as gt_config

    original_root_directory = gt_config.cache_settings["root_path"]
    working_dir = str(os.getcwd())

    # Compile on default
    p0 = OrchestratedProgam(backend, DaCeOrchestration.BuildAndRun)
    p0()
    assert os.path.exists(
        f"{working_dir}/.gt_cache_FV3_A/dacecache/"
        "test_caches_OrchestratedProgam___call__",
    ) or os.path.exists(
        f"{working_dir}/.gt_cache_FV3_A/dacecache/OrchestratedProgam___call__",
    )

    # Compile in another directory

    custom_path = f"{working_dir}/.my_cache_path"
    gt_config.cache_settings["root_path"] = custom_path
    p1 = OrchestratedProgam(backend, DaCeOrchestration.BuildAndRun)
    p1()
    assert os.path.exists(
        f"{custom_path}/.gt_cache_FV3_A/dacecache/"
        "test_caches_OrchestratedProgam___call__",
    ) or os.path.exists(
        f"{working_dir}/.gt_cache_FV3_A/dacecache/OrchestratedProgam___call__",
    )

    # Check relocability by copying the second cache directory,
    # changing the path of gt_config.cache_settings and trying to Run on it
    relocated_path = f"{working_dir}/.my_relocated_cache_path"
    shutil.copytree(custom_path, relocated_path, dirs_exist_ok=True)
    gt_config.cache_settings["root_path"] = relocated_path
    p2 = OrchestratedProgam(backend, DaCeOrchestration.Run)
    p2()

    # Generate a file exists error to check for bad path
    bogus_path = "./nope/notatall/nothappening"
    gt_config.cache_settings["root_path"] = bogus_path
    with pytest.raises(RuntimeError):
        OrchestratedProgam(backend, DaCeOrchestration.Run)

    # Restore cache settings
    gt_config.cache_settings["root_path"] = original_root_directory


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("dace:cpu"),
    ],
)
def test_relocatability(backend: str):
    import os
    import shutil

    import gt4py
    from gt4py.cartesian import config as gt_config

    from pace.util.mpi import MPI

    # Restore original dir name
    gt4py.cartesian.config.cache_settings["dir_name"] = os.environ.get(
        "GT_CACHE_DIR_NAME", f".gt_cache_{MPI.COMM_WORLD.Get_rank():06}"
    )

    backend_sanitized = backend.replace(":", "")

    # Compile on default
    p0 = OrchestratedProgam(backend, DaCeOrchestration.Python)
    p0()
    assert os.path.exists(
        f"./.gt_cache_000000/py38_1013/{backend_sanitized}/test_caches/_stencil/"
    )

    # Compile in another directory

    custom_path = "./.my_cache_path"
    gt_config.cache_settings["root_path"] = custom_path
    p1 = OrchestratedProgam(backend, DaCeOrchestration.Python)
    p1()
    assert os.path.exists(
        f"{custom_path}/.gt_cache_000000/py38_1013/{backend_sanitized}"
        "/test_caches/_stencil/"
    )

    # Check relocability by copying the second cache directory,
    # changing the path of gt_config.cache_settings and trying to Run on it
    relocated_path = "./.my_relocated_cache_path"
    shutil.copytree("./.gt_cache_000000", relocated_path, dirs_exist_ok=True)
    gt_config.cache_settings["root_path"] = relocated_path
    p2 = OrchestratedProgam(backend, DaCeOrchestration.Python)
    p2()
    assert os.path.exists(
        f"{relocated_path}/.gt_cache_000000/py38_1013/{backend_sanitized}"
        "/test_caches/_stencil/"
    )
