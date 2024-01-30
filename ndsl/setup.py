from typing import List

from setuptools import find_namespace_packages, setup


setup_requirements: List[str] = []

requirements = ["gt4py", "dace", "pace-util"]

test_requirements: List[str] = []


setup(
    author="NOAA/NASA",
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
    name="ndsl",
    license="BSD license",
    packages=find_namespace_packages(include=["ndsl.*"]),
    include_package_data=True,
    url="https://github.com/NOAA-GFDL/NDSL",
    version="0.0.0",
    zip_safe=False,
)
