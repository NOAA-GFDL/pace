import os
from pathlib import Path
from setuptools import find_namespace_packages, setup
from typing import List


def local_pkg(name: str, relative_path: str) -> str:
    """Returns an absolute path to a local package."""
    return f"{name} @ file://{Path(os.path.abspath(__file__)).parent / relative_path} "


requirements: List[str] = [
    "dacite",
    "f90nml",
    "numpy < 2.0.0", # numpy 2.x has breaking API changes
    "xarray",
    "zarr < 3.0.0", # zarr 3.x has breaking API changes
]

prod_requirements = [
    local_pkg("NDSL", "NDSL"),
    local_pkg("pyFV3", "pyFV3"),
    local_pkg("pySHiELD", "pySHiELD"),
]

test_requirements = [
    "mpi4py",
    "nbmake",
    "pytest",
]

lint_requirements = [
    "pre-commit"
]

extras_require = {
    "lint": lint_requirements,
    "prod": prod_requirements,
    "test": test_requirements,
}


setup(
    author="Allen Institute for AI",
    author_email="oliver.elbert@noaa.gov",
    python_requires=">=3.8,<3.12",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=requirements,
    extras_require=extras_require,
    name="pace",
    license="BSD license",
    packages=find_namespace_packages(include=["pace", "pace.*"]),
    include_package_data=True,
    url="https://github.com/NOAA-GFDL/pace",
    version="0.2.0",
    zip_safe=False,
)
