import os
from pathlib import Path
from typing import List

from setuptools import setup


def local_pkg(name: str, relative_path: str) -> str:
    """Returns an absolute path to a local package."""
    return f"{name} @ file://{Path(os.path.abspath(__file__)).parent / relative_path} "


requirements: List[str] = [
    local_pkg("ndsl", "NDSL"),
    local_pkg("pyFV3", "pyFV3"),
    local_pkg("pySHiELD", "pySHiELD"),
    "dacite",
    "f90nml",
    "numpy < 2.0.0",
    "xarray",
    "zarr < 3.0.0",
]

setup(install_requires=requirements)
