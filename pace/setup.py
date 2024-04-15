import os
from pathlib import Path
from typing import List

from setuptools import find_namespace_packages, setup


def local_pkg(name: str, relative_path: str) -> str:
    """Returns an absolute path to a local package."""
    path = f"{name} @ file://{Path(os.path.abspath(__file__)).parent / relative_path}"
    return path


setup_requirements: List[str] = []

requirements = [
    "ndsl",
    "dacite",
    "pyyaml",
    "mpi4py",
    "numpy",
    "netCDF4",
    "xarray",
    "zarr",
]

test_requirements: List[str] = []


setup(
    author="Allen Institute for AI",
    author_email="oliver.elbert@noaa.gov",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=requirements,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    name="pace",
    license="BSD license",
    packages=find_namespace_packages(include=["pace", "pace.*"]),
    include_package_data=True,
    url="https://github.com/NOAA-GFDL/pace",
    version="0.2.0",
    zip_safe=False,
)
