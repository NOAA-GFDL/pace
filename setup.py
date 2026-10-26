from pathlib import Path

from setuptools import setup


def local_pkg(name: str, relative_path: str, extras: str | None = None) -> str:
    """Returns an absolute path to a local package with optional extras."""
    path_str = f"{name} @ file://{Path(__file__).absolute().parent / relative_path}"
    if extras:
        return f"{name}[{extras}] @ file://{Path(__file__).absolute().parent / relative_path}"
    return path_str


requirements: list[str] = [
    local_pkg("ndsl", "NDSL", extras="pyfms"),
    local_pkg("pyfv3", "pyFV3"),
    local_pkg("pyshield", "pySHiELD"),
    "dacite",
    "f90nml",
    "numpy >= 2",
    "xarray",
    "zarr >= 3",
]

setup(install_requires=requirements)
