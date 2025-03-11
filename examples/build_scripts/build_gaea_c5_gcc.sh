#!/usr/bin/env bash

# Example bash script to install Pace to run bare-metal on Gaea's c5 cluster

set -e -x

# module load necessary system software
module use -a /usw/conda/modulefiles
module rm PrgEnv-intel
module load PrgEnv-gnu
module load gcc-native/12.3
module load boost/1.85.0
module load miniforge

export DACE_compiler_cpu_executable='which g++'
export GT4PY_EXTRA_COMPILE_ARGS="-O3 -ffast-math -fprefetch-loop-arrays -funroll-all-loops"
export OPENMP_CPPFLAGS="-fopenmp -fopenmp-simd"
export OPENMP_LDFLAGS="-fopenmp -fopenmp-simd"

# clone Pace and update submodules
git clone --recursive https://github.com/NOAA-GFDL/pace
cd pace

# create a conda environment for pace
conda create -y --name my_name python=3.11.7

# enter the environment and update it
conda activate my_name
pip3 install --upgrade pip setuptools wheel

# install the Pace dependencies, GT4Py, and Pace
pip3 install -r requirements_dev.txt -c constraints.txt
