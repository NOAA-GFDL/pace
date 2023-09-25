from typing import List, Optional, Tuple

from dace.sdfg import SDFG

import pace.util
from pace.dsl.caches.cache_location import get_cache_directory, get_cache_fullpath
from pace.dsl.dace.dace_config import DaceConfig, DaCeOrchestration


################################################
# Distributed compilation


def unblock_waiting_tiles(comm, sdfg_path: str) -> None:
    if comm and comm.Get_size() > 1:
        for tile in range(1, 6):
            tilesize = comm.Get_size() / 6
            comm.send(sdfg_path, dest=tile * tilesize + comm.Get_rank())


def build_info_filepath() -> str:
    return "build_info.txt"


def write_build_info(
    sdfg: SDFG, layout: Tuple[int, int], resolution_per_tile: List[int], backend: str
):
    """Write down all relevant information on the build to identify
    it at load time."""
    # Dev NOTE: we should be able to leverage sdfg.make_key to get a hash or
    # even go to a complete hash base system and read the data from the SDFG itself
    import os

    path_to_sdfg_dir = os.path.abspath(sdfg.build_folder)
    with open(f"{path_to_sdfg_dir}/{build_info_filepath()}", "w") as build_info_read:
        build_info_read.write("#Schema: Backend Layout Resolution per tile\n")
        build_info_read.write(f"{backend}\n")
        build_info_read.write(f"{str(layout)}\n")
        build_info_read.write(f"{str(resolution_per_tile)}\n")


################################################

################################################
# SDFG load (both .sdfg file and build directory containing .so)


def get_sdfg_path(
    daceprog_name: str,
    config: DaceConfig,
    sdfg_file_path: Optional[str] = None,
    override_run_only=False,
) -> Optional[str]:
    """Build an SDFG path from the qualified program name or it's direct path to .sdfg

    Args:
        program_name: qualified name in the form module_qualname if module is not locals
        sdfg_file_path: absolute path to a .sdfg file
    """
    import os

    # TODO: check DaceConfig for cache.strategy == name
    # Guarding against bad usage of this function
    if not override_run_only and config.get_orchestrate() != DaCeOrchestration.Run:
        return None

    # Case of a .sdfg file given by the user to be compiled
    if sdfg_file_path is not None:
        if not os.path.isfile(sdfg_file_path):
            raise RuntimeError(
                f"SDFG filepath {sdfg_file_path} cannot be found or is not a file"
            )
        return sdfg_file_path

    # Case of loading a precompiled .so - lookup using GT_CACHE
    cache_fullpath = get_cache_fullpath(config.code_path)
    sdfg_dir_path = f"{cache_fullpath}/dacecache/{daceprog_name}"
    if not os.path.isdir(sdfg_dir_path):
        raise RuntimeError(f"Precompiled SDFG is missing at {sdfg_dir_path}")

    # Check layout in build time matches layout now
    import ast

    with open(f"{sdfg_dir_path}/{build_info_filepath()}") as build_info_file:
        # Jump over schema comment
        build_info_file.readline()
        # Read in
        build_backend = build_info_file.readline().rstrip()
        if config.get_backend() != build_backend:
            raise RuntimeError(
                f"SDFG build for {build_backend}, {config._backend} has been asked"
            )
        # Check resolution per tile
        build_layout = ast.literal_eval(build_info_file.readline())
        build_resolution = ast.literal_eval(build_info_file.readline())
        if (config.tile_resolution[0] / config.layout[0]) != (
            build_resolution[0] / build_layout[0]
        ):
            raise RuntimeError(
                f"SDFG build for resolution {build_resolution}, "
                f"cannot be run with current resolution {config.tile_resolution}"
            )

    print(f"[DaCe Config] Rank {config.my_rank} loading SDFG {sdfg_dir_path}")

    return sdfg_dir_path


def set_distributed_caches(config: "DaceConfig"):
    """In Run mode, check required file then point current rank cache to source cache"""

    # Execute specific initialization per orchestration state
    orchestration_mode = config.get_orchestrate()
    if orchestration_mode == DaCeOrchestration.Python:
        return

    # Check that we have all the file we need to early out in case
    # of issues.
    if orchestration_mode == DaCeOrchestration.Run:
        import os

        cache_directory = get_cache_fullpath(config.code_path)
        if not os.path.exists(cache_directory):
            raise RuntimeError(
                f"{orchestration_mode} error: Could not find caches for rank "
                f"{config.my_rank} at {cache_directory}"
            )

    # Set read/write caches to the target rank
    from gt4py.cartesian import config as gt_config

    if config.do_compile:
        verb = "reading/writing"
    else:
        verb = "reading"

    gt_config.cache_settings["dir_name"] = get_cache_directory(config.code_path)
    pace.util.pace_log.info(
        f"[{orchestration_mode}] Rank {config.my_rank} "
        f"{verb} cache {gt_config.cache_settings['dir_name']}"
    )
