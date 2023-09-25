import enum
import os
from typing import Any, Dict, Optional, Tuple

import dace.config
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.frontend.python.parser import DaceProgram

from pace.dsl.caches.cache_location import identify_code_path
from pace.dsl.caches.codepath import FV3CodePath
from pace.dsl.gt4py_utils import is_gpu_backend
from pace.util._optional_imports import cupy as cp
from pace.util.communicator import Communicator, Partitioner


# This can be turned on to revert compilation for orchestration
# in a rank-compile-itself more, instead of the distributed top-tile
# mechanism.
DEACTIVATE_DISTRIBUTED_DACE_COMPILE = False


def _is_corner(rank: int, partitioner: Partitioner) -> bool:
    if partitioner.tile.on_tile_bottom(rank):
        if partitioner.tile.on_tile_left(rank):
            return True
        if partitioner.tile.on_tile_right(rank):
            return True
    if partitioner.tile.on_tile_top(rank):
        if partitioner.tile.on_tile_left(rank):
            return True
        if partitioner.tile.on_tile_right(rank):
            return True
    return False


def _smallest_rank_bottom(x: int, y: int, layout: Tuple[int, int]):
    return y == 0 and x == 1


def _smallest_rank_top(x: int, y: int, layout: Tuple[int, int]):
    return y == layout[1] - 1 and x == 1


def _smallest_rank_left(x: int, y: int, layout: Tuple[int, int]):
    return x == 0 and y == 1


def _smallest_rank_right(x: int, y: int, layout: Tuple[int, int]):
    return x == layout[0] - 1 and y == 1


def _smallest_rank_middle(x: int, y: int, layout: Tuple[int, int]):
    return layout[0] > 1 and layout[1] > 1 and x == 1 and y == 1


def _determine_compiling_ranks(
    config: "DaceConfig",
    partitioner: Partitioner,
) -> bool:
    """
    We try to map every layout to a 3x3 layout which MPI ranks
    looks like
        6 7 8
        3 4 5
        0 1 2
    Using the partitionner we find mapping of the given layout
    to all of those. For example on 4x4 layout
        12 13 14 15
        8  9  10 11
        4  5  6  7
        0  1  2  3
    therefore we map
        0 -> 0
        1 -> 1
        2 -> NOT COMPILING
        3 -> 2
        4 -> 3
        5 -> 4
        6 -> NOT COMPILING
        7 -> 5
        8 -> NOT COMPILING
        9 -> NOT COMPILING
        10 -> NOT COMPILING
        11 -> NOT COMPILING
        12 -> 6
        13 -> 7
        14 -> NOT COMPILING
        15 -> 8
    """

    # Tile 0 compiles
    if partitioner.tile_index(config.my_rank) != 0:
        return False

    # Corners compile
    if _is_corner(config.my_rank, partitioner):
        return True

    y, x = partitioner.tile.subtile_index(config.my_rank)

    # If edge or center tile, we give way to the smallest rank
    return (
        _smallest_rank_left(x, y, config.layout)
        or _smallest_rank_bottom(x, y, config.layout)
        or _smallest_rank_middle(x, y, config.layout)
        or _smallest_rank_right(x, y, config.layout)
        or _smallest_rank_top(x, y, config.layout)
    )


class DaCeOrchestration(enum.Enum):
    """
    Orchestration mode for DaCe

        Python: python orchestration
        Build: compile & save SDFG only
        BuildAndRun: compile & save SDFG, then run
        Run: load from .so and run, will fail if .so is not available
    """

    Python = 0
    Build = 1
    BuildAndRun = 2
    Run = 3


class FrozenCompiledSDFG:
    """
    Cache transform args to allow direct execution of the CSDFG

    Args:
        csdfg: compiled SDFG, e.g. loaded .so
        sdfg_args: transformed args to align for CSDFG direct execution

    WARNING: No checks are done on arguments, any memory swap (free/realloc)
    will lead to difficult to debug misbehavior
    """

    def __init__(
        self, daceprog: DaceProgram, csdfg: CompiledSDFG, args, kwargs
    ) -> None:
        self.csdfg = csdfg
        self.sdfg_args = daceprog._create_sdfg_args(csdfg.sdfg, args, kwargs)

    def __call__(self):
        return self.csdfg(**self.sdfg_args)


class DaceConfig:
    def __init__(
        self,
        communicator: Optional[Communicator],
        backend: str,
        tile_nx: int = 0,
        tile_nz: int = 0,
        orchestration: Optional[DaCeOrchestration] = None,
    ):
        # Recording SDFG loaded for fast re-access
        # ToDo: DaceConfig becomes a bit more than a read-only config
        #       with this. Should be refactor into a DaceExecutor carrying a config
        self.loaded_precompiled_SDFG: Dict[DaceProgram, FrozenCompiledSDFG] = {}

        # Temporary. This is a bit too out of the ordinary for the common user.
        # We should refactor the architecture to allow for a `gtc:orchestrated:dace:X`
        # backend that would signify both the `CPU|GPU` split and the orchestration mode
        if orchestration is None:
            fv3_dacemode_env_var = os.getenv("FV3_DACEMODE", "Python")
            # The below condition guard against defining empty FV3_DACEMODE and
            # awkward behavior of os.getenv returning "" even when not defined
            if fv3_dacemode_env_var is None or fv3_dacemode_env_var == "":
                fv3_dacemode_env_var = "Python"
            self._orchestrate = DaCeOrchestration[fv3_dacemode_env_var]
        else:
            self._orchestrate = orchestration

        # Debugging Dace orchestration deeper can be done by turning on `syncdebug`
        # We control this Dace configuration below with our own override
        dace_debug_env_var = os.getenv("PACE_DACE_DEBUG", "False") == "True"

        # Set the configuration of DaCe to a rigid & tested set of divergence
        # from the defaults when orchestrating
        if orchestration != DaCeOrchestration.Python:
            # Required to True for gt4py storage/memory
            dace.config.Config.set(
                "compiler",
                "allow_view_arguments",
                value=True,
            )
            # Removed --fmath
            dace.config.Config.set(
                "compiler",
                "cpu",
                "args",
                value="-std=c++14 -fPIC -Wall -Wextra -O3",
            )
            # Potentially buggy - deactivate
            dace.config.Config.set(
                "compiler",
                "cpu",
                "openmp_sections",
                value=0,
            )
            # Removed --fast-math
            dace.config.Config.set(
                "compiler",
                "cuda",
                "args",
                value="-std=c++14 -Xcompiler -fPIC -O3 -Xcompiler -march=native",
            )

            cuda_sm = 60
            if cp:
                cuda_sm = cp.cuda.Device(0).compute_capability
            dace.config.Config.set("compiler", "cuda", "cuda_arch", value=f"{cuda_sm}")
            # Block size/thread count is defaulted to an average value for recent
            # hardware (Pascal and upward). The problem of setting an optimized
            # block/thread is both hardware and problem dependant. Fine tuners
            # available in DaCe should be relied on for futher tuning of this value.
            dace.config.Config.set(
                "compiler", "cuda", "default_block_size", value="64,8,1"
            )
            # Potentially buggy - deactivate
            dace.config.Config.set(
                "compiler",
                "cuda",
                "max_concurrent_streams",
                value=-1,  # no concurrent streams, every kernel on defaultStream
            )
            # Speed up built time
            dace.config.Config.set(
                "compiler",
                "cuda",
                "unique_functions",
                value="none",
            )
            # Required for HaloEx callbacks and general code sanity
            dace.config.Config.set(
                "frontend",
                "dont_fuse_callbacks",
                value=True,
            )
            # Unroll all loop - outer loop should be exempted with dace.nounroll
            dace.config.Config.set(
                "frontend",
                "unroll_threshold",
                value=False,
            )
            # Allow for a longer stack dump when parsing fails
            dace.config.Config.set(
                "frontend",
                "verbose_errors",
                value=True,
            )
            # Build speed up by removing some deep copies
            dace.config.Config.set(
                "store_history",
                value=False,
            )

            # Enable to debug GPU failures
            dace.config.Config.set(
                "compiler", "cuda", "syncdebug", value=dace_debug_env_var
            )

        # attempt to kill the dace.conf to avoid confusion
        if dace.config.Config._cfg_filename:
            try:
                os.remove(dace.config.Config._cfg_filename)
            except OSError:
                pass

        self._backend = backend
        self.tile_resolution = [tile_nx, tile_nx, tile_nz]
        from pace.dsl.dace.build import set_distributed_caches

        # Distributed build required info
        if communicator:
            self.my_rank = communicator.rank
            self.rank_size = communicator.comm.Get_size()
            self.code_path = identify_code_path(self.my_rank, communicator.partitioner)
            self.layout = communicator.partitioner.layout
            self.do_compile = (
                DEACTIVATE_DISTRIBUTED_DACE_COMPILE
                or _determine_compiling_ranks(self, communicator.partitioner)
            )
        else:
            self.my_rank = 0
            self.rank_size = 1
            self.code_path = FV3CodePath.All
            self.layout = (1, 1)
            self.do_compile = True

        set_distributed_caches(self)

        if (
            self._orchestrate != DaCeOrchestration.Python
            and "dace" not in self._backend
        ):
            raise RuntimeError(
                "DaceConfig: orchestration can only be leverage "
                f"on dace or dace:gpu not on {self._backend}"
            )

    def is_dace_orchestrated(self) -> bool:
        return self._orchestrate != DaCeOrchestration.Python

    def is_gpu_backend(self) -> bool:
        return is_gpu_backend(self._backend)

    def get_backend(self) -> str:
        return self._backend

    def get_orchestrate(self) -> DaCeOrchestration:
        return self._orchestrate

    def get_sync_debug(self) -> bool:
        return dace.config.Config.get_bool("compiler", "cuda", "syncdebug")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "_orchestrate": str(self._orchestrate.name),
            "_backend": self._backend,
            "my_rank": self.my_rank,
            "rank_size": self.rank_size,
            "layout": self.layout,
            "tile_resolution": self.tile_resolution,
        }

    @classmethod
    def from_dict(cls, data: dict):
        config = cls(
            None,
            backend=data["_backend"],
            orchestration=DaCeOrchestration[data["_orchestrate"]],
        )
        config.my_rank = data["my_rank"]
        config.rank_size = data["rank_size"]
        config.layout = data["layout"]
        config.tile_resolution = data["tile_resolution"]
        return config
