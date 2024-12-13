[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache License][license-shield]][license-url]

# Pace

Pace is an implementation of the FV3GFS / SHiELD atmospheric model developed by NOAA/GFDL using the [NDSL](https://github.com/NOAA-GFDL/NDSL) middleware in Python, itself based on [GT4Py](https://github.com/GridTools/gt4py) and [DaCe](https://github.com/spcl/dace). The model can be run on a laptop using Python-based backend or on thousands of heterogeneous compute nodes of a large supercomputer.

🚧 **WARNING** This repo is under active development - supported features and procedures can change rapidly and without notice. 🚧

The repository model code is split between [pyFV3](https://github.com/NOAA-GFDL/pyFV3) for the dynamical core and [pySHiELD](https://github.com/NOAA-GFDL/pySHiELD) for the physics parametrization. A full depencies looks like the following:

```mermaid
flowchart TD
GT4Py.cartesian --> |Stencil DSL|NDSL
DaCe  --> |Full program opt|NDSL
NDSL --> pyFV3
NDSL --> pySHiELD
pyFV3 --> |Dynamics|Pace
pySHiELD --> |Physics|Pace

```

## Quickstart - bare metal

### Build

Pace requires:

- GCC > 9.2
- MPI
- Python 3.8.

For GPU backends CUDA and/or ROCm is required depending on the targeted hardware.

For GT stencils backends, you will also need the headers of the boost libraries in your `$PATH`. This could be down like this.

```shell
cd BOOST/ROOT
wget https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz
tar -xzf boost_1_79_0.tar.gz
mkdir -p boost_1_79_0/include
mv boost_1_79_0/boost boost_1_79_0/include/
export BOOST_ROOT=BOOST/ROOT/boost_1_79_0
```

When cloning Pace you will need to update the repository's submodules as well:

```shell
git clone --recursive https://github.com/NOAA-GFDL/pace.git
```

or if you have already cloned the repository:

```
git submodule update --init --recursive
```

We recommend creating a python `venv` or `conda` environment specifically for Pace.

```shell
python3 -m venv venv_name
source venv_name/bin/activate
```

Inside of your pace `venv` or conda environment pip install the Python requirements, GT4Py, and Pace:

```shell
pip3 install -r requirements_dev.txt -c constraints.txt
```

Shell scripts to install Pace on specific machines such as Gaea can be found in `examples/build_scripts/`.

### Run

Located in the directory `examples/configs/` there are several example configurations to exhibit the current functionality of Pace. We suggest a new user start with the `baroclinic_c12.yaml` configuration.

Before starting any run, including unit tests, the user must ensure that the proper input files are available. For the `baroclinic_c12.yaml` configuration a script to generate these files and place them in the location referenced in the configuration file is available in the `examples/` directory; `generate_eta_files.py`. To generate the files use the following commands from the top level of the clone of Pace:

```shell
mkdir tests/main/input
python3 examples/generate_eta_files.py
mv *eta*.nc tests/main/input
```
These commands will generate the files necessary and place them in the `tests/main/input` directory. Once the files are generated the `baroclinic_c12.yaml` configuration can be used to generate a run:

```shell
mpirun -n 6 python3 -m pace.run examples/configs/baroclinic_c12.yaml

# or with oversubscribe if you do not have at least 6 cores
mpirun -n 6 --oversubscribe python3 -m pace.run examples/configs/baroclinic_c12.yaml
```

After the run completes, you will see an output direcotry `output.zarr`. An example to visualize the output is provided in `examples/plot_output.py`. See the [driver example](examples/README.md) section for more details.

### Environment variable configuration

- `PACE_CONSTANTS`: Pace is bundled with various constants.
  - `GFDL` NOAA's FV3 dynamical core constants (original port)
  - `GFS` Constant as defined in NOAA GFS
  - `GEOS`  Constant as defined in GEOS v13
- `PACE_FLOAT_PRECISION`: default precision of the field & scalars in the numerics. Default to 64.
- `PACE_LOGLEVEL`: logging level to display (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default to INFO.

## Quickstart - Docker

### Build

While it is possible to install and build pace bare-metal, we can ensure all system libraries are installed with the correct versions by using a Docker container to test and develop pace.

First, you will need to update the git submodules so that any dependencies are cloned and at the correct version:

```shell
git submodule update --init --recursive
```

Then build the `pace` docker image at the top level.

```shell
make build
```

### Run

```shell
make dev
mpirun --mca btl_vader_single_copy_mechanism none -n 6 python -m pace.run /examples/configs/baroclinic_c12.yaml
```

## History

This repository was first developed at [AI2](https://github.com/ai2cm/pace) and the institute conserves an archived copy with the latest state before the NOAA took over.

[contributors-shield]: https://img.shields.io/github/contributors/NOAA-GFDL/pace.svg
[contributors-url]: https://github.com/NOAA-GFDL/pace/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/NOAA-GFDL/pace.svg
[stars-url]: https://github.com/NOAA-GFDL/pace/stargazers
[issues-shield]: https://img.shields.io/github/issues/NOAA-GFDL/pace.svg
[issues-url]: https://github.com/NOAA-GFDL/pace/issues
[license-shield]: https://img.shields.io/github/license/NOAA-GFDL/pace.svg
[license-url]: https://github.com/NOAA-GFDL/pace/blob/main/LICENSE.md

## Running pace in containers
Docker images exist in the Github Container Registry associated with the NOAA-GFDL organization.
These images are publicly accessible and can be used to run a Docker container to work with pace.
The following are directions on how to setup the pace conda environment interactively in a container.

The latest images can be pulled with the Docker as shown below or
with any other container management tools:

```shell
docker pull ghcr.io/noaa-gfdl/pace_mpich:3.8
```
for MPICH installation of MPI; and
```shell
docker pull ghcr.io/noaa-gfdl/pace_openmpi:3.8
```
for OpenMPI installation of MPI.

If permission issues arise during the pull, a Github personal token
may be required.  The steps to create a personal token is found
[here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)

Once the token has been generated, the image can be pulled for example with with:
```shell
docker login --username GITHUB_USERNAME --password TOKEN
docker pull ghcr.io/noaa-gfdl/pace_mpich:3.8
```

Any container management tools compatible with Docker images can be used
to run the container interactively from the pulled image.
With Docker, the following command runs the container interactively.
```shell
docker run -it pace_mpich:3.8
```

In the container, the default `base` conda environment is already activated.
The `pace` conda environment can be created by following the steps below:

```shell
git clone --recursive -b develop https://github.com/NOAA-GFDL/pace.git pace
cd pace
cp /home/scripts/setup_env.sh . && chmod +x setup_env.sh
source ./setup_env.sh
```
