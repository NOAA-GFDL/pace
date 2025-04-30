#!/usr/bin/env bash
# Requires you have git and python>=3.8 with venv and pip installed
# Requires an MPI library for mpi4py, such as libopenmpi3 and libmopenmpi3-dev

set -e -x

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python3 -m venv venv
. venv/bin/activate

rundir=$(pwd)
cd ${SCRIPT_DIR}/../

pip install -c constraints-dev.txt -e .

deactivate
cd $rundir
