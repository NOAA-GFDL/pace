import os
from pathlib import Path
from typing import List

from setuptools import setup


def find_pkg(name: str, relative_path: str) -> str:
    """
    Returns an absolute path to a local package or
    provides location of Github repository.
    """
    dependecy_path = f"{Path(os.path.abspath(__file__)).parent / relative_path}"
    if os.path.isdir(dependecy_path):
        return (
            f"{name} @ file://{Path(os.path.abspath(__file__)).parent / relative_path} "
        )
    else:
        return f"{name} @ git+https://github.com/NOAA-GFDL/{name}.git@develop"


requirements: List[str] = [
    find_pkg("NDSL", "NDSL"),
    find_pkg("pyFV3", "pyFV3"),
    find_pkg("pySHiELD", "pySHiELD"),
    "dacite",
    "f90nml",
    "numpy < 2.0.0",
    "xarray",
    "zarr < 3.0.0",
]

setup(install_requires=requirements)
