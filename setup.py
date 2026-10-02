from pathlib import Path

from setuptools import setup


def local_pkg(name: str, relative_path: str, extras: str = "") -> str:
    """Returns an absolute path to a local package."""
    extras_spec = f"[{extras}]" if extras else ""
    return f"{name}{extras_spec} @ file://{Path(__file__).absolute().parent / relative_path} "


requirements: list[str] = [
    local_pkg("ndsl", "NDSL", extras="pyfms"),
    local_pkg("pyfv3", "pyFV3"),
    local_pkg("pyshield", "pySHiELD"),
    "dacite",
    "f90nml",
    "numpy >= 2",
    "xarray",
    "zarr < 3.0.0",
]

setup(install_requires=requirements)
