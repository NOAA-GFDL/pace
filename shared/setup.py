import sys
from typing import List

from setuptools import find_namespace_packages, setup


setup_requirements: List[str] = []

requirements = ["gt4py", "fv3gfs-util"]
if sys.version_info.major == 3 and sys.version_info.minor == 6:
    requirements.append("dataclasses")

test_requirements: List[str] = []


setup(
    author="Allen Institute for AI",
    author_email="elynnw@allenai.org",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=requirements,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    name="pace-shared",
    license="BSD license",
    packages=find_namespace_packages(include=["pace.stencils.*"]),
    include_package_data=True,
    url="https://github.com/ai2cm/pace",
    version="0.1.0",
    zip_safe=False,
)
