# Benchmarking README

The tests contained in this archive are for benchmarking purposes only.  Any
distribution beyond those personnel performing the tests need explicit approval
from NOAA/GFDL (Seth Underwood or Rusty Benson).

## Cloning benchmark repository and generating conda environment

Pace requires GCC > 9.2, MPI, and Python 3.8 on your system, and CUDA is required to run with a GPU backend. You will also need the headers of the boost libraries in your `$PATH` (boost itself does not need to be installed).

```shell
cd BOOST/ROOT
wget https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz
tar -xzf boost_1_79_0.tar.gz
mkdir -p boost_1_79_0/include
mv boost_1_79_0/boost boost_1_79_0/include/
export BOOST_ROOT=BOOST/ROOT/boost_1_79_0
```

To clone the benchmark branch use the command:

```shell
git clone --recursive -b benchmark git@github.com:NOAA-GFDL/pace.git
```

or if you have already cloned the repository:

```shell
git submodule update --init --recursive
```

After cloning, change into the directory containing the clone. To generate the conda environment use the following commands:

```shell
conda create -y --name <desired_name> python=3.8
conda activate <desired_name>
pip3 install --upgrade pip setuptools wheel
pip3 install -r requirements_dev.txt -c constraints.txt
```

## Benchmarking configurations

There are four configurations of the PACE application contained within the branch to be used for benchmarking:

```shell
driver/examples/configs/baroclinic_c384_cpu.yaml
driver/examples/configs/baroclinic_c384_gpu.yaml
driver/examples/configs/baroclinic_c3072_cpu.yaml
driver/examples/configs/baroclinic_c3072_gpu.yaml
```

## Building

To build with the DaCe backends, set the following environment variables:

```shell
FV3_DACEMODE=Build
PACE_FLOAT_PRECISION=64
PACE_LOGLEVEL=INFO
PYTHONOPTIMIZE=1
OMP_NUM_THREAD=1
```

Adjust the time of the configuration to be built such that the time of the build is for one timestep. For example:

```shell
dt_atmos: 450
seconds: 450
```
## Running
To build with the DaCe backends, set the following environment variables:

```shell
FV3_DACEMODE=Run
PACE_FLOAT_PRECISION=64
PACE_LOGLEVEL=INFO
PYTHONOPTIMIZE=1
OMP_NUM_THREAD=1
```

Adjust the time of the configuration to be run to the desired length, example:

```shell
dt_atmos: 450
days: 9
```

The time for the build or run can be set with units of seconds, minutes, hours, or days.

An example command to start the build or run process with MPI using the DaCe CPU backend for the c384 configuration:

```shell
mpirun -n 1536 python3 -m pace.driver.run driver/examples/configs/baroclinic_c384_cpu.yaml
```

The build or run requires 1536 ranks, given that layout of 16x16 ranks per tile, and there are 6 tiles.
