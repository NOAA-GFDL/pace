#!/bin/bash

source /root/.bashrc
conda activate pace

chmod +x /home/scripts
mkdir tests/main/input
/home/scripts/create_eta79.py && mv eta79.nc tests/main/input
/home/scripts/create_eta91.py && mv eta91.nc tests/main/input

pytest -x tests/main
