#!/usr/bin/env bash f

# Example bash script to install Pace to run bare-metal on Gaea's c4 cluster

set -e -x

# module load necessary system software
module rm PrgEnv-intel
module load PrgEnv-gnu
module rm gcc
module load gcc/12.2.0
module load boost/1.79.0
module load python/3.9

# clone Pace and update submodules
git clone --recursive https://github.com/NOAA-GFDL/pace
cd pace

# create a conda environment for pace
conda create -y --name my_name python=3.8

# enter the environment and update it
conda activate my_name
pip3 install --upgrade pip setuptools wheel

# install the Pace dependencies, GT4Py, and Pace
pip3 install -r requirements_dev.txt -c constraints.txt
