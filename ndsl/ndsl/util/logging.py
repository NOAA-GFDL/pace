import logging
import os
import sys

from mpi4py import MPI


LOGLEVEL = os.environ.get("PACE_LOGLEVEL", "INFO").upper()

# Python log levels are hierarchical, therefore setting INFO
# means DEBUG and everything lower will be logged.
AVAILABLE_LOG_LEVELS = {
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def _pace_logger():
    name_log = logging.getLogger(__name__)
    name_log.setLevel(LOGLEVEL)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(LOGLEVEL)
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
