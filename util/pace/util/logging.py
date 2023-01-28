import logging
import sys

from mpi4py import MPI


def _pace_logger():
    name_log = logging.getLogger(__name__)
    name_log.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt=(
            f"%(asctime)s|%(levelname)s|rank {MPI.COMM_WORLD.Get_rank()}|"
            "%(name)s:%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    name_log.addHandler(handler)
    return name_log


pace_log = _pace_logger()
