from pace.dsl.caches.codepath import FV3CodePath
from pace.util import Partitioner


def identify_code_path(
    rank: int,
    partitioner: Partitioner,
) -> FV3CodePath:
    if partitioner.layout == (1, 1) or partitioner.layout == [1, 1]:
        return FV3CodePath.All
    elif partitioner.layout[0] == 1 or partitioner.layout[1] == 1:
        raise NotImplementedError(
            f"Build for layout {partitioner.layout} is not handled"
        )
    else:
        if partitioner.tile.on_tile_bottom(rank):
            if partitioner.tile.on_tile_left(rank):
                return FV3CodePath.BottomLeft
            if partitioner.tile.on_tile_right(rank):
                return FV3CodePath.BottomRight
            else:
                return FV3CodePath.Bottom
        if partitioner.tile.on_tile_top(rank):
            if partitioner.tile.on_tile_left(rank):
                return FV3CodePath.TopLeft
            if partitioner.tile.on_tile_right(rank):
                return FV3CodePath.TopRight
            else:
                return FV3CodePath.Top
        else:
            if partitioner.tile.on_tile_left(rank):
                return FV3CodePath.Left
            if partitioner.tile.on_tile_right(rank):
                return FV3CodePath.Right
            else:
                return FV3CodePath.Center


def get_cache_fullpath(code_path: FV3CodePath) -> str:
    from gt4py.cartesian import config as gt_config

    return f"{gt_config.cache_settings['root_path']}/.gt_cache_{code_path}"


def get_cache_directory(code_path: FV3CodePath) -> str:
    return f".gt_cache_{code_path}"
